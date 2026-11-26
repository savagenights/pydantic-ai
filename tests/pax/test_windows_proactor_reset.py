"""Windows ProactorEventLoop socket teardown suppresses WinError 10054 without hiding other errors."""

from __future__ import annotations

import sys
from asyncio import proactor_events
from collections.abc import Callable
from typing import cast

import pytest

from pydantic_ai_pax.__main__ import quiet_windows_proactor_reset


class _DummySock:
    def __init__(
        self,
        *,
        shutdown_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.shutdown_error = shutdown_error
        self.close_error = close_error
        self.closed = False

    def shutdown(self, _how: int) -> None:
        if self.shutdown_error is not None:
            raise self.shutdown_error

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _DummyProtocol:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.error = error
        self.calls: list[BaseException | None] = []

    def connection_lost(self, exc: BaseException | None) -> None:
        self.calls.append(exc)
        if self.error is not None:
            raise self.error


class _DummyServer:
    def __init__(self) -> None:
        self.detached: list[object] = []

    def _detach(self, transport: object) -> None:
        self.detached.append(transport)


class _FakeTransport:
    def __init__(
        self,
        *,
        sock: _DummySock | None,
        protocol: _DummyProtocol,
        server: object = None,
        called_connection_lost: bool = False,
    ) -> None:
        self._called_connection_lost = called_connection_lost
        self._sock = sock
        self._protocol = protocol
        self._server = server

    @property
    def sock(self) -> _DummySock | None:
        return self._sock

    @property
    def server(self) -> object:
        return self._server

    @property
    def called_connection_lost(self) -> bool:
        return self._called_connection_lost

    def _call_connection_lost(self, exc: BaseException | None) -> None:
        if self._called_connection_lost:
            return
        try:
            self._protocol.connection_lost(exc)
        finally:
            assert self._sock is not None
            self._sock.shutdown(2)
            self._sock.close()
            self._sock = None
            if self._server is not None:
                detach = cast(Callable[[object], None], getattr(self._server, '_detach'))
                detach(self)
                self._server = None
            self._called_connection_lost = True


def _win_reset_error(winerror: int = 10054) -> ConnectionResetError:
    err = ConnectionResetError(winerror, f'simulated winerror {winerror}')
    setattr(err, 'winerror', winerror)
    return err


def _invoke_call_connection_lost(
    transport: _FakeTransport,
    exc: BaseException | None = None,
    transport_cls: type[object] = _FakeTransport,
) -> None:
    method = cast(
        Callable[[_FakeTransport, BaseException | None], None],
        getattr(transport_cls, '_call_connection_lost'),
    )
    method(transport, exc)


def _patch_proactor(
    monkeypatch: pytest.MonkeyPatch,
    platform: str = 'win32',
    transport_cls: object = _FakeTransport,
) -> None:
    monkeypatch.setattr(sys, 'platform', platform)
    monkeypatch.setattr(
        proactor_events,
        '_ProactorBasePipeTransport',
        transport_cls,
        raising=False,
    )


def test_normal_teardown_passes_through_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_proactor(monkeypatch)
    sock = _DummySock()
    protocol = _DummyProtocol()
    server = _DummyServer()
    transport = _FakeTransport(sock=sock, protocol=protocol, server=server)

    with quiet_windows_proactor_reset():
        _invoke_call_connection_lost(transport)

    assert protocol.calls == [None]
    assert sock.closed is True
    assert transport.sock is None
    assert server.detached == [transport]
    assert transport.server is None
    assert transport.called_connection_lost is True


def test_suppresses_winerror_10054_on_shutdown_and_finishes_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_proactor(monkeypatch)
    sentinel = RuntimeError('peer closed')
    sock = _DummySock(shutdown_error=_win_reset_error())
    protocol = _DummyProtocol()
    server = _DummyServer()
    transport = _FakeTransport(sock=sock, protocol=protocol, server=server)

    with quiet_windows_proactor_reset():
        _invoke_call_connection_lost(transport, sentinel)
        _invoke_call_connection_lost(transport, sentinel)

    assert protocol.calls == [sentinel]
    assert sock.closed is True
    assert transport.sock is None
    assert server.detached == [transport]
    assert transport.server is None
    assert transport.called_connection_lost is True


def test_preserves_protocol_error_when_shutdown_also_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_proactor(monkeypatch)
    callback_error = RuntimeError('callback failed')
    sock = _DummySock(shutdown_error=_win_reset_error())
    protocol = _DummyProtocol(error=callback_error)
    server = _DummyServer()
    transport = _FakeTransport(sock=sock, protocol=protocol, server=server)

    with quiet_windows_proactor_reset(), pytest.raises(RuntimeError, match='callback failed') as exc_info:
        _invoke_call_connection_lost(transport)

    assert exc_info.value is callback_error
    assert sock.closed is True
    assert transport.sock is None
    assert server.detached == [transport]
    assert transport.server is None
    assert transport.called_connection_lost is True


def test_does_not_reraise_outer_exception_context_when_protocol_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_proactor(monkeypatch)
    sock = _DummySock(shutdown_error=_win_reset_error())
    transport = _FakeTransport(sock=sock, protocol=_DummyProtocol())

    with quiet_windows_proactor_reset():
        try:
            raise ValueError('outer active exception')
        except ValueError:
            _invoke_call_connection_lost(transport)

    assert sock.closed is True
    assert transport.sock is None
    assert transport.called_connection_lost is True


def test_cleanup_continues_when_socket_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_proactor(monkeypatch)
    sock = _DummySock(
        shutdown_error=_win_reset_error(),
        close_error=OSError('close failed'),
    )
    server = _DummyServer()
    transport = _FakeTransport(sock=sock, protocol=_DummyProtocol(), server=server)

    with quiet_windows_proactor_reset():
        _invoke_call_connection_lost(transport)

    assert sock.closed is True
    assert transport.sock is None
    assert server.detached == [transport]
    assert transport.server is None
    assert transport.called_connection_lost is True


def test_reraises_non_10054_connection_reset_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_proactor(monkeypatch)
    sock = _DummySock(shutdown_error=_win_reset_error(10053))
    transport = _FakeTransport(sock=sock, protocol=_DummyProtocol())

    with quiet_windows_proactor_reset(), pytest.raises(ConnectionResetError) as exc_info:
        _invoke_call_connection_lost(transport)

    assert getattr(exc_info.value, 'winerror', None) == 10053


def test_reraises_winerror_10054_from_protocol_connection_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_proactor(monkeypatch)
    sock = _DummySock()
    transport = _FakeTransport(
        sock=sock,
        protocol=_DummyProtocol(error=_win_reset_error()),
    )

    with quiet_windows_proactor_reset(), pytest.raises(ConnectionResetError) as exc_info:
        _invoke_call_connection_lost(transport)

    assert getattr(exc_info.value, 'winerror', None) == 10054
    assert sock.closed is True
    assert transport.called_connection_lost is True


def _plain_os_error_10054() -> OSError:
    err = OSError('plain oserror')
    setattr(err, 'winerror', 10054)
    return err


@pytest.mark.parametrize(
    ('called_connection_lost', 'has_sock', 'error_factory'),
    [
        (True, True, lambda: _win_reset_error()),
        (False, False, lambda: _win_reset_error()),
        (False, True, _plain_os_error_10054),
    ],
)
def test_disqualifying_guard_conditions_reraise_without_recovery(
    monkeypatch: pytest.MonkeyPatch,
    called_connection_lost: bool,
    has_sock: bool,
    error_factory: Callable[[], OSError],
) -> None:
    raised = error_factory()

    class _DirectRaiseTransport:
        @staticmethod
        def _call_connection_lost(_transport: _FakeTransport, _exc: BaseException | None) -> None:
            raise raised

    _patch_proactor(monkeypatch, transport_cls=_DirectRaiseTransport)
    sock = _DummySock() if has_sock else None
    server = _DummyServer()
    transport = _FakeTransport(
        sock=sock,
        protocol=_DummyProtocol(),
        server=server,
        called_connection_lost=called_connection_lost,
    )

    with quiet_windows_proactor_reset(), pytest.raises(OSError) as exc_info:
        _invoke_call_connection_lost(transport, transport_cls=_DirectRaiseTransport)

    assert exc_info.value is raised
    assert server.detached == []
    assert transport.called_connection_lost is called_connection_lost
    if sock is not None:
        assert sock.closed is False
        assert transport.sock is sock


def test_restores_original_method_on_normal_and_exceptional_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FreshTransport(_FakeTransport):
        pass

    _patch_proactor(monkeypatch, transport_cls=_FreshTransport)
    original = cast(object, getattr(_FreshTransport, '_call_connection_lost'))

    with quiet_windows_proactor_reset():
        assert cast(object, getattr(_FreshTransport, '_call_connection_lost')) is not original
    assert cast(object, getattr(_FreshTransport, '_call_connection_lost')) is original

    with pytest.raises(RuntimeError, match='body boom'), quiet_windows_proactor_reset():
        assert cast(object, getattr(_FreshTransport, '_call_connection_lost')) is not original
        raise RuntimeError('body boom')
    assert cast(object, getattr(_FreshTransport, '_call_connection_lost')) is original


def test_noop_on_non_windows_and_missing_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_proactor(monkeypatch, 'linux')
    original = cast(object, getattr(_FakeTransport, '_call_connection_lost'))

    with quiet_windows_proactor_reset():
        assert cast(object, getattr(_FakeTransport, '_call_connection_lost')) is original

    _patch_proactor(monkeypatch, 'win32', transport_cls=None)
    with quiet_windows_proactor_reset():
        assert getattr(proactor_events, '_ProactorBasePipeTransport') is None


def test_handles_python_311_zero_arg_server_detach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ZeroArgServer:
        def __init__(self) -> None:
            self.detached_calls = 0

        def _detach(self) -> None:
            self.detached_calls += 1

    _patch_proactor(monkeypatch)
    sock = _DummySock(shutdown_error=_win_reset_error())
    server = _ZeroArgServer()
    transport = _FakeTransport(sock=sock, protocol=_DummyProtocol(), server=server)

    with quiet_windows_proactor_reset():
        _invoke_call_connection_lost(transport)

    assert sock.closed is True
    assert transport.sock is None
    assert server.detached_calls == 1
    assert transport.server is None
    assert transport.called_connection_lost is True
