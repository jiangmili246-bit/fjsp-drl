import os
import sys
sys.path.append(os.path.abspath('.'))

from utils.dynamic_instance_generator import build_dynamic_instance_from_fjs_file


def test_dynamic_labels():
    f = 'data_dev/1005/10j_5m_001.fjs'
    urg = build_dynamic_instance_from_fjs_file(f, scenario='urgent_insertion', seed=11)['job_meta']
    rnd = build_dynamic_instance_from_fjs_file(f, scenario='random_arrival', seed=11)['job_meta']
    con = build_dynamic_instance_from_fjs_file(f, scenario='concurrent', seed=11)['job_meta']

    assert any(w == 4.0 for w in urg['priority_weight']), 'urgent job w_i=4 not found'
    assert any(c == 1 for c in rnd['arrival_type']), 'random arrival c_i=1 not found'
    assert any(c == 2 for c in con['arrival_type']), 'urgent insertion c_i=2 not found in concurrent'


def test_feature_contract_by_source():
    src = open('env/fjsp_env.py', 'r', encoding='utf-8').read()
    assert 'self.feat_opes_batch[:, 0, :] = k_ij' in src
    assert 'self.feat_opes_batch[:, 1, :] = est' in src
    assert 'self.feat_opes_batch[:, 2, :] = ept' in src
    assert 'self.feat_opes_batch[:, 3' not in src, 'operation features should not contain extra dims like DD/w copies'
    assert 'machine AT must not change due to arrival' in src


if __name__ == '__main__':
    test_dynamic_labels()
    test_feature_contract_by_source()
    print('Phase4 feature contract test passed.')
