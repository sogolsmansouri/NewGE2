#!/usr/bin/env python3
"""Non-overlapping GE2 host phases and independent CUDA activity accounting."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

PHASES = {
    'Sampling': ('dataloader.negative_sample',),
    'Shuffling': ('active_edges.shuffle',),
    'Mapping': ('dataloader.edge_remap', 'graph.global_to_local_map', 'graph.edge_remap', 'trainer.dense_map'),
    'Model/update host range': ('trainer.model_train', 'trainer.embedding_update'),
    'Movement/lookup host range': ('dataloader.boundary.parameter_swap', 'graph.edge_h2d',
                                   'trainer.gpu_parameter_load', 'dataloader.load_cpu_parameters'),
}


def counter_phases(epoch):
    total = epoch['official_epoch_ms']/1000
    if total <= 0:
        raise ValueError('Nonpositive epoch duration')
    rows = []
    for phase, metrics in PHASES.items():
        # Missing instrumentation must not silently become a zero-cost operation.
        values = [epoch['metrics'][m]['elapsed_ms']/1000 for m in metrics]
        if any(v < 0 for v in values):
            raise ValueError('Negative phase time')
        rows.append(dict(phase=phase, seconds=sum(values), metrics=list(metrics)))
    residual = total-sum(r['seconds'] for r in rows)
    if residual < -0.002:
        raise ValueError('Counter phases exceed epoch wall time')
    rows.append(dict(phase='Graph construction/control/other', seconds=max(0,residual), metrics=['residual']))
    for row in rows:
        row['percent'] = 100*row['seconds']/total
    return dict(epoch_seconds=total, phases=rows, idle_seconds=None,
                warning='Host elapsed ranges include CUDA waits; residual is NOT GPU idle.')


def union(intervals):
    result = []
    for start,end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0],max(end,result[-1][1]))
        else:
            result.append((start,end))
    return result


def phase_segments(ranges, bounds):
    changes = defaultdict(list)
    changes[bounds[0]]; changes[bounds[1]]
    names = {name:phase for phase,items in PHASES.items() for name in items}
    for a,b,name in ranges:
        if name not in names:
            continue
        a,b = max(a,bounds[0]),min(b,bounds[1])
        if b > a:
            changes[a].append((names[name],1)); changes[b].append((names[name],-1))
    active = defaultdict(int)
    last = bounds[0]
    result = []
    for t in sorted(changes):
        if t > last:
            phases = [p for p,n in active.items() if n]
            if len(phases) > 1:
                raise ValueError('Unexpected overlapping host phases: '+repr(phases))
            result.append((last,t,phases[0] if phases else 'Graph construction/control/other'))
        for phase,delta in changes[t]:
            active[phase] += delta
        last = t
    return result


def device_accounting(connection, bounds, host_segments):
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    changes = defaultdict(list)
    changes[bounds[0]]; changes[bounds[1]]
    services = {}
    found = set()
    for table,kind in [('CUPTI_ACTIVITY_KIND_KERNEL','kernel'),
                       ('CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL','kernel'),
                       ('CUPTI_ACTIVITY_KIND_MEMCPY','copy'),('CUPTI_ACTIVITY_KIND_MEMSET','memset')]:
        if table not in tables:
            continue
        query = 'SELECT start,end'+(',copyKind,bytes' if kind=='copy' else '')+' FROM '+table+' WHERE start < ? AND end > ?'
        seconds = 0; count = 0
        for record in connection.execute(query,(bounds[1],bounds[0])):
            a,b = max(record[0],bounds[0]),min(record[1],bounds[1])
            if b <= a:
                continue
            changes[a].append((kind,1)); changes[b].append((kind,-1))
            count += 1; seconds += (b-a)/1e9
            if kind=='copy':
                direction = {1:'H2D',2:'D2H',8:'D2D'}.get(record[2],'other_copy')
                row = services.setdefault(direction,dict(seconds=0,bytes=0,calls=0,clipped_calls=0))
                row['seconds'] += (b-a)/1e9; row['calls'] += 1
                if a==record[0] and b==record[1]:
                    row['bytes'] += record[3]
                else:
                    row['clipped_calls'] += 1
        if count:
            found.add(kind)
    if 'kernel' not in found:
        raise ValueError('Missing CUDA kernel trace; cannot infer idle')
    active = defaultdict(int)
    totals = defaultdict(float)
    gaps = []
    last = bounds[0]
    for t in sorted(changes):
        if t > last:
            live = [k for k,n in active.items() if n]
            label = '+'.join(sorted(live)) if live else 'no recorded CUDA work'
            totals[label] += (t-last)/1e9
            if not live:
                gaps.append((last,t))
        for kind,delta in changes[t]:
            active[kind] += delta
        last = t
    idle_by_phase = defaultdict(float)
    j = 0
    for a,b,phase in host_segments:
        while j < len(gaps) and gaps[j][1] <= a:
            j += 1
        k = j
        while k < len(gaps) and gaps[k][0] < b:
            idle_by_phase[phase] += max(0,min(b,gaps[k][1])-max(a,gaps[k][0]))/1e9
            k += 1
    return dict(window_seconds=(bounds[1]-bounds[0])/1e9, activity_seconds=dict(totals),
                gpu_inactive_by_host_phase=dict(idle_by_phase), copy_service_nonadditive=services,
                missing_empty_tables=sorted({'CUPTI_ACTIVITY_KIND_MEMSET','CUPTI_ACTIVITY_KIND_MEMCPY'}-tables),
                warning='No recorded CUDA work is trace inactivity, not proof of hardware power-idle or its cause.')


def trace_summary(path, state_index=None):
    with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as conn:
        base = "SELECT n.start,n.end,n.globalTid FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE COALESCE(n.text,s.value)=? AND n.end IS NOT NULL ORDER BY n.start"
        if state_index is None:
            marks = list(conn.execute(base,('ge2.profile.native_epoch',)))
            if len(marks) != 1:
                raise ValueError('Require exactly one complete profiled native epoch')
            a,b,tid = marks[0]; scope='whole_native_epoch_2'
        else:
            if state_index < 0:
                raise ValueError('State index must be nonnegative')
            marks = list(conn.execute(base,('dataloader.boundary.total',)))
            if len(marks) <= state_index+1:
                raise ValueError('Requested cycle not captured')
            a,_,tid = marks[state_index]; b=marks[state_index+1][0]; scope='bounded_state_cycle_only'
        bounds = (a,b)
        rows = conn.execute("SELECT n.start,n.end,COALESCE(n.text,s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE n.globalTid=? AND n.end IS NOT NULL AND n.start < ? AND n.end > ?",(tid,b,a))
        segments = phase_segments(rows,bounds)
        host = defaultdict(float)
        for start,end,phase in segments:
            host[phase] += (end-start)/1e9
        device = device_accounting(conn,bounds,segments)
        result=dict(scope=scope,start_ns=a,end_ns=b,main_tid=tid,host_phase_seconds=dict(host),**device)
        if state_index is None:
            final=list(conn.execute(base,('ge2.profile.epoch_finalize',)))
            if len(final)!=1 or final[0][0]<b or final[0][2]!=tid:
                raise ValueError('Missing finalization/drain range after epoch')
            result['post_timer_finalize_seconds']=(final[0][1]-final[0][0])/1e9
        return result


def plot(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    phases=report['counters']['phases']
    trace=report.get('trace')
    duration=report['counters']['epoch_seconds']
    if trace and trace['scope']=='whole_native_epoch_2':
        duration=trace['window_seconds']
        phases=[dict(phase=row['phase'],percent=100*trace['host_phase_seconds'].get(row['phase'],0)/duration)
                for row in phases]
    fig,axes=plt.subplots(2 if trace else 1,1,figsize=(7.0,4.5 if trace else 2.6),squeeze=False)
    ax=axes[0,0]
    labels=['Sampling','Shuffling','Mapping','Model/update','Movement/lookup','Other']
    colors=['#387ba8','#c68426','#39917f','#994c6e','#d15e4a','#8d9498']
    left=0
    for row,label,color in zip(phases,labels,colors):
        ax.barh([0],[row['percent']],left=left,color=color,label=label,height=.45)
        if row['percent']>=5:
            ax.text(left+row['percent']/2,0,f"{row['percent']:.1f}%",ha='center',va='center',color='white',fontsize=8)
        left+=row['percent']
    ax.set(xlim=(0,100),yticks=[],xlabel='Instrumented host-epoch wall time (%)',
           title=f"{report['label']}\nEpoch {report['epoch']}: {duration:.3f} s (host phases)")
    ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.51),frameon=False,fontsize=8)
    if trace:
        ax=axes[1,0]; left=0
        palette={'kernel':'#994c6e','copy':'#d15e4a','memset':'#d0ab56','no recorded CUDA work':'#d4d7da'}
        for name,seconds in sorted(trace['activity_seconds'].items()):
            pct=100*seconds/trace['window_seconds']
            ax.barh([0],[pct],left=left,height=.45,color=palette.get(name,'#39917f'),label=name)
            if pct>=5:
                ax.text(left+pct/2,0,f'{pct:.1f}%',ha='center',va='center',fontsize=8)
            left+=pct
        scope='Full profiled epoch' if trace['scope']=='whole_native_epoch_2' else 'Separate bounded cycle, NOT full epoch'
        ax.set(xlim=(0,100),yticks=[],xlabel='Traced device-window wall time (%)',title=f"{scope}: {trace['window_seconds']:.3f} s")
        ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.51),frameon=False,fontsize=8)
    for ax in axes[:,0]:
        ax.spines[['top','right','left']].set_visible(False)
    fig.subplots_adjust(top=.87,bottom=.17,hspace=1.9)
    fig.savefig(out/'motivation.pdf',bbox_inches='tight')
    fig.savefig(out/'motivation.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--counter-log',type=Path,required=True)
    p.add_argument('--epoch',type=int,default=2)
    p.add_argument('--sqlite',type=Path)
    p.add_argument('--state-index',type=int)
    p.add_argument('--label',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--plot',action='store_true')
    args=p.parse_args()
    from summarize_local_tw_ge2_analysis import parse_counter_log
    epoch=parse_counter_log(args.counter_log)[args.epoch]
    result=dict(label=args.label,epoch=args.epoch,counter_log=str(args.counter_log.resolve()),
                counter_log_sha256=hashlib.sha256(args.counter_log.read_bytes()).hexdigest(),
                counters=counter_phases(epoch),parameter_h2d_bytes=epoch['parameter_h2d_bytes'],
                parameter_d2h_bytes=epoch['parameter_d2h_bytes'],graph_h2d_bytes=epoch['graph_h2d_bytes'])
    if args.sqlite:
        result['trace']=trace_summary(args.sqlite.resolve(),args.state_index)
        result['trace_sqlite']=str(args.sqlite.resolve())
        result['trace_state_index']=args.state_index
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'breakdown.json').write_text(json.dumps(result,indent=2)+'\n')
    with (args.out/'host_phases.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['phase','seconds','percent'])
        for row in result['counters']['phases']:
            w.writerow([row[k] for k in ('phase','seconds','percent')])
    with (args.out/'operation_counters.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['metric','calls','inclusive_elapsed_s','mean_ms_per_call'])
        for name,row in sorted(epoch['metrics'].items()):
            w.writerow([name,row['calls'],row['elapsed_ms']/1000,
                        row['elapsed_ms']/row['calls'] if row['calls'] else ''])
    if args.plot:
        plot(result,args.out)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
