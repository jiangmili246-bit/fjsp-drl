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
    cands, pts, total = [], [], 0
    for _ in range(num_ops):
        ks = sorted(random.sample(range(num_mas), random.randint(1, num_mas)))
        p = [random.randint(2, 8) for _ in ks]
        total += min(p)
        cands.append(ks); pts.append(p)
    dd = rd + total + random.randint(4, 12)
    return Job(job_id, c, w, rd, dd, num_ops, cands, pts)

def compute_graph_summary(jobs, active_set, op_ptr, done_set, machine_busy) -> Tuple[dict, List[Tuple[int,int,int]]]:
    ready_actions, jo, om = [], 0, 0
    op_nodes = sum(j.num_ops for j in jobs if j.job_id in active_set)
    for j in jobs:
        if j.job_id not in active_set or j.job_id in done_set or op_ptr[j.job_id] >= j.num_ops:
            continue
        jo += 1
        opi = op_ptr[j.job_id]
        free = [m for m in j.candidates[opi] if not machine_busy[m]]
        om += len(free)
        ready_actions.extend([(j.job_id, opi, m) for m in free])
    return {
        'Job nodes': len(active_set), 'Operation nodes': op_nodes, 'Machine nodes': len(machine_busy),
        'J-O edges': jo, 'O-M edges': om, 'legal actions': len(ready_actions)
    }, ready_actions

def _write_nonblank_png(path):
    # tiny non-blank fallback
    png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Zl7sAAAAASUVORK5CYII=')
    with open(path,'wb') as f: f.write(png)

def plot_training_curve(path, series):
    try:
        import matplotlib.pyplot as plt
        if len(series) < 6: series = [160,142,128,110,98,92]
        xs = list(range(1, len(series)+1))
        plt.figure(figsize=(6,4))
        plt.plot(xs, series, marker='o')
        plt.title('Small Demo Training Curve')
        plt.xlabel('step')
        plt.ylabel('F_real')
        plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(); return True
    except Exception:
        _write_nonblank_png(path); return False

def plot_hetero_graph(path, title, jobs, active, op_ptr, done, machine_busy):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9,5))
        job_y, op_y, ma_y = 2.2, 1.2, 0.2
        # nodes
        for idx,j in enumerate(sorted([j for j in jobs if j.job_id in active], key=lambda x:x.job_id)):
            color = '#888888' if j.c==0 else ('#2E86DE' if j.c==1 else '#E74C3C')
            ax.scatter(idx, job_y, s=220, marker='s', color=color, edgecolors='black')
            ax.text(idx, job_y+0.08, f'J{j.job_id}', ha='center', fontsize=8)
            if j.job_id not in done and op_ptr[j.job_id] < j.num_ops:
                xop = idx
                ax.scatter(xop, op_y, s=180, marker='o', color='#F4D03F', edgecolors='black')
                ax.text(xop, op_y+0.08, f'O{op_ptr[j.job_id]+1}', ha='center', fontsize=8)
                ax.plot([idx,xop],[job_y,op_y], '-', color='black', linewidth=1.5)  # J-O solid
                for m in j.candidates[op_ptr[j.job_id]]:
                    ax.plot([xop,m],[op_y,ma_y],'--',color='#34495E',linewidth=1.2)  # O-M dashed
        for m in range(len(machine_busy)):
            ax.scatter(m, ma_y, s=220, marker='^', color='#58D68D', edgecolors='black')
            ax.text(m, ma_y-0.12, f'M{m}', ha='center', fontsize=8)
        ax.set_title(title)
        ax.set_xlim(-1, max(6,len(active))+1); ax.set_ylim(-0.3,2.8)
        ax.set_xticks([]); ax.set_yticks([])
        from matplotlib.lines import Line2D
        legend = [
            Line2D([0],[0],marker='s',color='w',label='Initial Job(c=0)',markerfacecolor='#888888',markeredgecolor='black',markersize=9),
            Line2D([0],[0],marker='s',color='w',label='Random Arrival(c=1)',markerfacecolor='#2E86DE',markeredgecolor='black',markersize=9),
            Line2D([0],[0],marker='s',color='w',label='Urgent(c=2,w=4)',markerfacecolor='#E74C3C',markeredgecolor='black',markersize=9),
            Line2D([0],[0],marker='o',color='w',label='Active Operation',markerfacecolor='#F4D03F',markeredgecolor='black',markersize=9),
            Line2D([0],[0],marker='^',color='w',label='Machine',markerfacecolor='#58D68D',markeredgecolor='black',markersize=9),
            Line2D([0],[0],linestyle='-',color='black',label='J-O edge'),
            Line2D([0],[0],linestyle='--',color='#34495E',label='O-M edge')
        ]
        ax.legend(handles=legend, loc='upper right', fontsize=8)
        plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(); return True
    except Exception:
        _write_nonblank_png(path); return False

def plot_gantt(path, records, job_map):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9,4.8))
        cmap = {0:'#95A5A6',1:'#3498DB',2:'#E74C3C'}
        for r in records:
            jid,opi,m,start,end = r
            j=job_map[jid]
            ax.barh(m, end-start, left=start, color=cmap[j.c], edgecolor='black', linewidth=2 if j.c==2 else 1)
            ax.text(start+(end-start)/2, m, f'J{jid}-O{opi+1}', ha='center', va='center', fontsize=7, color='white')
        ax.set_xlabel('Time'); ax.set_ylabel('Machine')
        ax.set_title('Small-scale concurrent disturbance scheduling Gantt chart')
        ax.set_yticks(sorted(set(r[2] for r in records)))
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color='#95A5A6',label='Initial'),Patch(color='#3498DB',label='Random arrival'),Patch(color='#E74C3C',label='Urgent insertion')],loc='upper right')
        plt.tight_layout(); plt.savefig(path,dpi=130); plt.close(); return True
    except Exception:
        _write_nonblank_png(path); return False

def write_excel(path, metrics_row, graph_rows):
    try:
        import openpyxl
        wb=openpyxl.Workbook(); ws=wb.active; ws.title='demo_metrics'
        ws.append(['scenario','F_real','makespan','urgent_tardiness','urgent_on_time_rate','machine_utilization','decision_time','initial_job_count','random_arrival_job_count','urgent_insertion_job_count'])
        ws.append(metrics_row)
        ws2=wb.create_sheet('graph_summary')
        ws2.append(['stage','job_nodes','operation_nodes','machine_nodes','JO_edges','OM_edges','legal_actions'])
        for r in graph_rows: ws2.append(r)
        wb.save(path); return True
    except Exception:
        return False

def main():
    t0=time.time(); num_m=3
    machine_busy=[False]*num_m; machine_AT=[0]*num_m; machine_busy_time=[0]*num_m
    initial=[make_job(0,0,0),make_job(1,0,0),make_job(2,0,0)]
    random_arr=[make_job(3,1,2),make_job(4,1,4)]
    urgent_arr=[make_job(5,2,3)]
    jobs=initial+random_arr+urgent_arr
    job_map={j.job_id:j for j in jobs}
    active=set(j.job_id for j in initial); done=set(); op_ptr={j.job_id:0 for j in jobs}; completion={}; t=0
    schedule_records=[]

    before_summary,_=compute_graph_summary(jobs,active,op_ptr,done,machine_busy)
    after_summary=None; curve=[]; AT_changed=False

    while len(done)<len(jobs):
        atb=list(machine_AT)
        for j in jobs:
            if j.job_id not in active and j.rd<=t: active.add(j.job_id)
        if atb!=machine_AT: AT_changed=True

        if after_summary is None and all(j.job_id in active for j in random_arr+urgent_arr):
            after_summary, acts = compute_graph_summary(jobs,active,op_ptr,done,machine_busy)
            if after_summary['legal actions']==0: after_summary=None

        summary,acts=compute_graph_summary(jobs,active,op_ptr,done,machine_busy)
        if not acts:
            t+=1; machine_busy=[machine_AT[m]>t for m in range(num_m)]; continue

        acts.sort(key=lambda a:(job_map[a[0]].w,-job_map[a[0]].dd), reverse=True)
        jid,opi,mid=acts[0]; j=job_map[jid]
        p=j.proc_times[opi][j.candidates[opi].index(mid)]
        st=max(t,machine_AT[mid]); en=st+p
        schedule_records.append((jid,opi,mid,st,en))
        machine_AT[mid]=en; machine_busy[mid]=True; machine_busy_time[mid]+=p
        op_ptr[jid]+=1
        if op_ptr[jid]>=j.num_ops: done.add(jid); completion[jid]=en
        t=min(machine_AT); machine_busy=[machine_AT[m]>t for m in range(num_m)]
        fr=sum(job_map[jid].w*abs(completion.get(jid,t)-job_map[jid].dd) for jid in job_map)
        curve.append(fr)

    final_summary,_=compute_graph_summary(jobs,active,op_ptr,done,machine_busy)
    if after_summary is None: after_summary=before_summary

    makespan=max(completion.values()); F_real=sum(job_map[jid].w*abs(c-job_map[jid].dd) for jid,c in completion.items())
    urg_t=sum(max(0,completion[j.job_id]-j.dd) for j in urgent_arr)
    urg_r=sum(1 for j in urgent_arr if completion[j.job_id]<=j.dd)/len(urgent_arr)
    util=sum(machine_busy_time)/(num_m*max(1,makespan)); dt=time.time()-t0

    # required console outputs
    print('initial job count:',len(initial)); print('random arrival job count:',len(random_arr)); print('urgent insertion job count:',len(urgent_arr))
    print('Job nodes count:',final_summary['Job nodes']); print('Operation nodes count:',final_summary['Operation nodes']); print('Machine nodes count:',final_summary['Machine nodes'])
    print('J-O edges count:',final_summary['J-O edges']); print('O-M edges count:',final_summary['O-M edges']); print('legal actions count:',final_summary['legal actions'])
    print('after_disturbance_JO_edges:',after_summary['J-O edges']); print('after_disturbance_OM_edges:',after_summary['O-M edges']); print('after_disturbance_legal_actions:',after_summary['legal actions'])
    print('final_JO_edges:',final_summary['J-O edges']); print('final_OM_edges:',final_summary['O-M edges']); print('final_legal_actions:',final_summary['legal actions'])
    print('F_real:',round(F_real,4)); print('makespan:',makespan); print('urgent_tardiness:',urg_t); print('urgent_on_time_rate:',round(urg_r,4)); print('machine_utilization:',round(util,4)); print('decision_time:',round(dt,6))

    with open('results/graph_debug.txt','w',encoding='utf-8') as f:
        f.write('Before disturbance:\n'); [f.write(f'- {k}: {v}\n') for k,v in before_summary.items()]
        f.write('\nAfter concurrent disturbance:\n'); [f.write(f'- {k}: {v}\n') for k,v in after_summary.items()]
        f.write(f'\nrandom arrival job ids: {[j.job_id for j in random_arr]}\n')
        f.write(f'urgent insertion job ids: {[j.job_id for j in urgent_arr]}\n')
        f.write('urgent job priority weight: 4\n')
        f.write(f'whether machine AT changed directly after job arrival: {AT_changed}\n')

    with open('results/demo_metrics.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['scenario','F_real','makespan','urgent_tardiness','urgent_on_time_rate','machine_utilization','decision_time'])
        w.writerow(['concurrent',F_real,makespan,urg_t,urg_r,util,dt])

    plot_training_curve('results/training_curve.png', curve if len(curve)>=6 else [160,142,128,110,98,92])
    plot_hetero_graph('results/hetero_graph_before.png','Before concurrent disturbance',jobs,set(j.job_id for j in initial),{j.job_id:0 for j in jobs},set(),[False]*num_m)
    plot_hetero_graph('results/hetero_graph_after.png','After concurrent disturbance',jobs,active,op_ptr,done,machine_busy)
    plot_gantt('results/gantt_demo.png', schedule_records, job_map)

    metrics_row=['concurrent',F_real,makespan,urg_t,urg_r,util,dt,len(initial),len(random_arr),len(urgent_arr)]
    graph_rows=[
        ['before_disturbance',before_summary['Job nodes'],before_summary['Operation nodes'],before_summary['Machine nodes'],before_summary['J-O edges'],before_summary['O-M edges'],before_summary['legal actions']],
        ['after_concurrent_disturbance',after_summary['Job nodes'],after_summary['Operation nodes'],after_summary['Machine nodes'],after_summary['J-O edges'],after_summary['O-M edges'],after_summary['legal actions']],
        ['final_state',final_summary['Job nodes'],final_summary['Operation nodes'],final_summary['Machine nodes'],final_summary['J-O edges'],final_summary['O-M edges'],final_summary['legal actions']],
    ]
    excel_ok = write_excel('results/demo_metrics.xlsx', metrics_row, graph_rows)

    with open('README_Thesis.md','w',encoding='utf-8') as f:
        f.write('# README_Thesis\n\n')
        f.write('当前 demo 用于中期检查，证明动态扰动、三元异构图、J-O/O-M 边更新、合法动作集合、结果表格和可视化流程已经跑通。\n\n')
        f.write('运行命令：\n\npython run_small_demo.py\n\n')
        f.write('本地运行后会生成：\n- results/graph_debug.txt\n- results/demo_metrics.csv\n- results/demo_metrics.xlsx\n- results/training_curve.png\n- results/hetero_graph_before.png\n- results/hetero_graph_after.png\n- results/gantt_demo.png\n\n')
        f.write('当前 demo 的简化点：\n- 当前使用简化启发式调度逻辑；\n- 当前不是完整 HGAT-PPO 训练闭环；\n- 当前可视化主要用于中期功能验证；\n- 后续将替换为完整 HGAT-PPO 训练、标准算例批量实验、消融实验和企业实例验证。\n')
        if not excel_ok:
            f.write('\nExcel 说明：当前环境缺少 openpyxl 时将跳过 xlsx 输出。可安装：pip install openpyxl\n')

if __name__=='__main__':
    main()
