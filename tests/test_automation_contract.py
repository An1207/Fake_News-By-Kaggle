import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

from news_api.schemas import AutomationInput

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_preview_is_disabled_and_never_calls_network(capsys):
    spec = importlib.util.spec_from_file_location('automation_client', ROOT / 'scripts/automation_client.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = json.loads((ROOT / 'integrations/make/pipeline.json').read_text(encoding='utf-8'))
    assert contract['enabled'] is False and contract['optional_gpt']['enabled'] is False
    assert all(scenario['enabled'] is False for scenario in contract['scenarios'])
    with patch.object(module, 'build_opener', side_effect=AssertionError('preview must stay offline')):
        assert module.main([]) == 0
    assert json.loads(capsys.readouterr().out)['mode'] == 'offline_preview'
    assert AutomationInput.model_validate_json((ROOT / 'integrations/make/request.example.json').read_text(encoding='utf-8')).request_id
