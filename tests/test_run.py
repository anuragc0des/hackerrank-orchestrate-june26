import logging
from pathlib import Path

import pytest

import run


def test_setup_logging_creates_logs_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    called = {}

    def fake_basicConfig(*args, **kwargs):
        called['kwargs'] = kwargs

    monkeypatch.setattr(logging, 'basicConfig', fake_basicConfig)

    assert not (tmp_path / 'logs').exists()
    run.setup_logging()

    assert (tmp_path / 'logs').is_dir()
    assert 'kwargs' in called
    handlers = called['kwargs'].get('handlers', [])
    assert len(handlers) == 2
    file_handler = handlers[1]
    assert hasattr(file_handler, 'baseFilename')
    assert Path(file_handler.baseFilename).resolve() == (tmp_path / 'logs' / 'pipeline.log').resolve()
