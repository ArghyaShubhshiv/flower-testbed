#!/usr/bin/env bash
# Low-noise measurement settings for any Linux box (Intel or AMD). Not persistent.
#
#   sudo bash hwsig/prep_host.sh            apply (turbo/boost disabled)
#   sudo bash hwsig/prep_host.sh --boost    apply, but leave turbo/boost on
#   sudo bash hwsig/prep_host.sh --restore  undo
set -uo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
STATE=/tmp/hwsig_prep_state.sh
CPU=/sys/devices/system/cpu

if [[ "${1:-}" == "--restore" ]]; then
  [[ -f $STATE ]] && bash "$STATE" 2>/dev/null && rm -f "$STATE" && echo restored || echo "nothing to restore"
  exit 0
fi

save() { [[ -f "$1" ]] && echo "echo '$(cat "$1")' > '$1'"; }
{
  for k in perf_event_paranoid nmi_watchdog sched_schedstats; do
    f=/proc/sys/kernel/$k; [[ -f $f ]] && echo "sysctl -qw kernel.$k=$(cat $f)"; done
  for f in $CPU/cpu*/cpufreq/scaling_governor $CPU/cpu*/cpufreq/energy_performance_preference \
           $CPU/cpufreq/boost $CPU/intel_pstate/no_turbo; do save "$f"; done
  systemctl is-active --quiet irqbalance 2>/dev/null && echo "systemctl start irqbalance"
  for z in /sys/class/powercap/*/energy_uj; do [[ -f $z ]] && echo "chmod $(stat -c %a "$z") '$z'"; done
} > "$STATE"

# counters: per-CPU + kernel events; free the NMI-watchdog counter; scheduler stats
sysctl -qw kernel.perf_event_paranoid=-1
sysctl -qw kernel.nmi_watchdog=0 2>/dev/null
sysctl -qw kernel.sched_schedstats=1 2>/dev/null
# energy: RAPL readable without root; MSR driver for AMD per-core energy
for z in /sys/class/powercap/*/energy_uj; do [[ -f $z ]] && chmod a+r "$z"; done
modprobe msr 2>/dev/null
# fixed frequency policy
for f in $CPU/cpu*/cpufreq/scaling_governor; do echo performance > "$f" 2>/dev/null; done
for f in $CPU/cpu*/cpufreq/energy_performance_preference; do echo performance > "$f" 2>/dev/null; done
if [[ "${1:-}" != "--boost" ]]; then
  [[ -f $CPU/cpufreq/boost ]] && echo 0 > $CPU/cpufreq/boost
  [[ -f $CPU/intel_pstate/no_turbo ]] && echo 1 > $CPU/intel_pstate/no_turbo
fi
# device interrupts onto CPU 0 and its SMT sibling (the server/monitor core)
HK=$(cat $CPU/cpu0/topology/thread_siblings_list 2>/dev/null || echo 0)
systemctl stop irqbalance 2>/dev/null
for d in /proc/irq/*/; do echo "$HK" > "$d/smp_affinity_list" 2>/dev/null; done

echo "applied (undo: sudo bash $0 --restore). IRQs -> cpu $HK"
