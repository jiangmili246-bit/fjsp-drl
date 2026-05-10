import os


def main():
    src_env = open('env/fjsp_env.py', 'r', encoding='utf-8').read()
    src_ppo = open('PPO_model.py', 'r', encoding='utf-8').read()

    assert 'def build_legal_actions' in src_env
    assert 'Illegal action: operation' in src_env
    assert 'Illegal action: machine' in src_env
    assert 'job has not arrived' in src_env or 'has not arrived' in src_env
    assert 'legal_action_mask_batch' in src_env

    assert 'legal_mask = state.legal_action_mask_batch' in src_ppo
    assert '[DEBUG] ready operations:' in src_ppo
    assert '[DEBUG] legal actions:' in src_ppo
    assert '[DEBUG] action mask shape:' in src_ppo
    assert '[DEBUG] selected action:' in src_ppo

    print('Phase5 action-mask contract test passed.')


if __name__ == '__main__':
    main()
