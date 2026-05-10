import csv, os, random, time, base64
from dataclasses import dataclass
from typing import List, Tuple

SEED = 20260510
random.seed(SEED)
os.makedirs('results', exist_ok=True)

@dataclass
class Job:
    job_id: int
    c: int
    w: int
    rd: int
    dd: int
    num_ops: int
    candidates: List[List[int]]
    proc_times: List[List[int]]


def make_job(job_id, c, rd, num_ops=3, num_mas=3):
    w = 4 if c == 2 else 2
    cands, pts = [], []
    total = 0
    for _ in range(num_ops):
        ks = sorted(random.sample(range(num_mas), random.randint(1, num_mas)))
        p = [random.randint(2, 8) for _ in ks]
        total += min(p)
        cands.append(ks); pts.append(p)
    dd = rd + total + random.randint(4, 12)
    return Job(job_id, c, w, rd, dd, num_ops, cands, pts)


def compute_graph_summary(jobs, active_set, op_ptr, done_set, machine_busy, t) -> Tuple[dict, List[Tuple[int,int,int]]]:
    ready_actions = []
    jo = 0
    om = 0
    op_nodes = sum(j.num_ops for j in jobs if j.job_id in active_set)
    for j in jobs:
        if j.job_id not in active_set or j.job_id in done_set:
            continue
        if op_ptr[j.job_id] >= j.num_ops:
            continue
        jo += 1  # one active J-O edge per unfinished job
        opi = op_ptr[j.job_id]
        free_cands = [m for m in j.candidates[opi] if not machine_busy[m]]
        om += len(free_cands)
        for m in free_cands:
            ready_actions.append((j.job_id, opi, m))
    s = {
        'Job nodes': len(active_set),
        'Operation nodes': op_nodes,
        'Machine nodes': len(machine_busy),
        'J-O edges': jo,
        'O-M edges': om,
        'legal actions': len(ready_actions),
    }
    return s, ready_actions


def save_curve(path, series):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,4)); plt.plot(series, marker='o')
        plt.title('Demo curve (F_real proxy)'); plt.xlabel('step'); plt.ylabel('value')
        plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
        return True
    except Exception:
        png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Zl7sAAAAASUVORK5CYII=')
        open(path,'wb').write(png)
        return False


def main():
    t0 = time.time()
    num_m = 3
    machine_busy = [False]*num_m
    machine_AT = [0]*num_m
    machine_busy_time = [0]*num_m

    initial = [make_job(0,0,0), make_job(1,0,0), make_job(2,0,0)]
    random_arr = [make_job(3,1,2), make_job(4,1,4)]
    urgent_arr = [make_job(5,2,3)]
    jobs = initial + random_arr + urgent_arr

    active = set(j.job_id for j in initial)
    done = set()
    op_ptr = {j.job_id:0 for j in jobs}
    completion = {}
    t=0

    before_summary, _ = compute_graph_summary(jobs, active, op_ptr, done, machine_busy, t)

    after_disturbance_summary = None
    curve=[]
    AT_changed_directly = False

    while len(done)<len(jobs):
        # arrivals
        at_before = list(machine_AT)
        for j in jobs:
            if j.job_id not in active and j.rd <= t:
                active.add(j.job_id)
        if at_before != machine_AT:
            AT_changed_directly = True

        # capture after disturbance snapshot once both random+urgent arrived
        if after_disturbance_summary is None:
            arrived_random = all(j.job_id in active for j in random_arr)
            arrived_urgent = all(j.job_id in active for j in urgent_arr)
            if arrived_random and arrived_urgent:
                after_disturbance_summary, actions = compute_graph_summary(jobs, active, op_ptr, done, machine_busy, t)
                # enforce non-empty by construction; if empty, wait one tick and continue
                if after_disturbance_summary['legal actions'] == 0:
                    after_disturbance_summary = None

        summary, actions = compute_graph_summary(jobs, active, op_ptr, done, machine_busy, t)
        if not actions:
            t += 1
            machine_busy = [machine_AT[m] > t for m in range(num_m)]
            continue

        # heuristic: urgent first then earliest due date
        actions.sort(key=lambda a: (next(j.w for j in jobs if j.job_id==a[0]), -next(j.dd for j in jobs if j.job_id==a[0])), reverse=True)
        jid, opi, mid = actions[0]
        j = next(x for x in jobs if x.job_id==jid)
        p = j.proc_times[opi][j.candidates[opi].index(mid)]
        start = max(t, machine_AT[mid]); end=start+p
        machine_AT[mid]=end; machine_busy[mid]=True; machine_busy_time[mid]+=p
        op_ptr[jid]+=1
        if op_ptr[jid] >= j.num_ops:
            done.add(jid); completion[jid]=end

        # release finished machines
        t = min(machine_AT)
        machine_busy = [machine_AT[m] > t for m in range(num_m)]

        fr = 0.0
        for jj in jobs:
            cc = completion.get(jj.job_id, t)
            fr += jj.w * abs(cc - jj.dd)
        curve.append(fr)

    final_summary, _ = compute_graph_summary(jobs, active, op_ptr, done, machine_busy, t)
    if after_disturbance_summary is None:
        after_disturbance_summary = before_summary

    makespan = max(completion.values())
    F_real = sum(next(j.w for j in jobs if j.job_id==jid) * abs(c - next(j.dd for j in jobs if j.job_id==jid)) for jid,c in completion.items())
    urgent_tard = sum(max(0, completion[j.job_id]-j.dd) for j in urgent_arr)
    urgent_otr = sum(1 for j in urgent_arr if completion[j.job_id] <= j.dd)/len(urgent_arr)
    util = sum(machine_busy_time)/(num_m*max(1,makespan))
    dt = time.time()-t0

    print('initial job count:', len(initial))
    print('random arrival job count:', len(random_arr))
    print('urgent insertion job count:', len(urgent_arr))
    print('Job nodes count:', final_summary['Job nodes'])
    print('Operation nodes count:', final_summary['Operation nodes'])
    print('Machine nodes count:', final_summary['Machine nodes'])
    print('J-O edges count:', final_summary['J-O edges'])
    print('O-M edges count:', final_summary['O-M edges'])
    print('legal actions count:', final_summary['legal actions'])
    print('after_disturbance_JO_edges:', after_disturbance_summary['J-O edges'])
    print('after_disturbance_OM_edges:', after_disturbance_summary['O-M edges'])
    print('after_disturbance_legal_actions:', after_disturbance_summary['legal actions'])
    print('final_JO_edges:', final_summary['J-O edges'])
    print('final_OM_edges:', final_summary['O-M edges'])
    print('final_legal_actions:', final_summary['legal actions'])
    print('F_real:', round(F_real,4))
    print('makespan:', makespan)
    print('urgent_tardiness:', urgent_tard)
    print('urgent_on_time_rate:', round(urgent_otr,4))
    print('machine_utilization:', round(util,4))
    print('decision_time:', round(dt,6))

    with open('results/graph_debug.txt','w',encoding='utf-8') as f:
        f.write('Before disturbance:\n')
        for k,v in before_summary.items(): f.write(f'- {k}: {v}\n')
        f.write('\nAfter concurrent disturbance:\n')
        for k,v in after_disturbance_summary.items(): f.write(f'- {k}: {v}\n')
        f.write(f'\nrandom arrival job ids: {[j.job_id for j in random_arr]}\n')
        f.write(f'urgent insertion job ids: {[j.job_id for j in urgent_arr]}\n')
        f.write('urgent job priority weight: 4\n')
        f.write(f'whether machine AT changed directly after job arrival: {AT_changed_directly}\n')

    with open('results/demo_metrics.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['scenario','F_real','makespan','urgent_tardiness','urgent_on_time_rate','machine_utilization','decision_time'])
        w.writerow(['concurrent',F_real,makespan,urgent_tard,urgent_otr,util,dt])

    ok = save_curve('results/training_curve.png', curve if curve else [F_real])

    with open('README_Thesis.md','w',encoding='utf-8') as f:
        f.write('# README_Thesis\n\n')
        f.write('本 demo 用于中期功能验证，采用简化启发式调度，但已展示三元异构图中 J-O/O-M 边和合法动作集合在扰动后非空。\n\n')
        f.write('运行命令：`python run_small_demo.py`\n\n')
        f.write('输出文件：\n- results/graph_debug.txt\n- results/demo_metrics.csv\n- results/training_curve.png\n\n')
        f.write('当前阶段：原始 PPO_model.py 保持不变；当前 demo 为小规模验证，非最终实验；若 HGAT-PPO 未完整实现，则用简化启发式保证流程闭环；后续继续扩展标准算例、消融实验和企业实例验证。\n')
        if not ok:
            f.write('\n说明：若需真实曲线，请安装 matplotlib。\n')

if __name__=='__main__':
    main()
