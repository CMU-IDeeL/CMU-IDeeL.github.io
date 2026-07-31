#!/usr/bin/env bash
# =============================================================================
#  CMU 11-785 — Introduction to Deep Learning, Fall 2026
#  Recitation: Introduction to Linux & Infrastructure Essentials
#  DELIVERABLE 2: linux_masterclass.sh
#  Interactive CLI Playground for Students
#
#  HOW TO USE THIS SCRIPT:
#    Do NOT run it blindly with: bash linux_masterclass.sh
#    Instead, read each section, understand the annotated explanation,
#    and run commands ONE BY ONE in your terminal.
#    This is a teaching document, not a deployment script.
#
#  COMPATIBLE: Bash 4.x+, tested on Ubuntu 20.04/22.04, CentOS 7/8,
#              PSC Bridges-2 (Red Hat Enterprise Linux 8.x)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0: ORIENTATION — UNDERSTANDING YOUR SHELL ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
#
# Theory:
#   When you SSH into a remote Linux machine, you land inside a shell process.
#   The shell (typically bash) is an interactive command interpreter. Every
#   command you type is forked as a child process of that shell, runs, and
#   returns an exit code. Exit code 0 = success. Non-zero = failure.
#   The shell exposes your context through ENVIRONMENT VARIABLES: $HOME, $PATH,
#   $USER, $SHELL, and dozens more. These are inherited by every child process.

echo "=========================================================="
echo " CMU 11-785 Linux CLI Masterclass — Begin Orientation"
echo "=========================================================="

# Print WHO you are and WHERE you are
echo ""
echo "--- Identity & Location ---"
whoami          # prints the username of the current effective user
echo "You are: $(whoami)"

hostname        # prints the machine's network hostname
echo "Machine hostname: $(hostname)"

pwd             # Print Working Directory — your current location in the FHS
echo "Current directory (pwd): $(pwd)"

echo ""
echo "--- Shell & Environment ---"
echo "Your shell binary: $SHELL"
echo "Your home directory: $HOME"
echo "Your PATH (where binaries are searched): $PATH"

# The PATH variable is a colon-separated list of directories.
# When you type 'python3', the shell searches each PATH directory left-to-right
# and runs the first matching executable binary it finds.
echo ""
echo "PATH directories (one per line):"
echo "$PATH" | tr ':' '\n'
# tr replaces all ':' characters with newline characters — simple field splitting


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: FILE SYSTEM HIERARCHY — NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
#
# Theory:
#   Linux uses a single unified directory tree rooted at '/'. Unlike Windows,
#   there are no drive letters (C:, D:). Every file on every mounted disk or
#   network share appears somewhere under '/'. The path separator is '/' not '\'.
#
#   Key FHS directories every DL practitioner must know:
#     /home/<user>/   — your personal workspace (limited quota on HPC!)
#     /scratch/       — high-speed scratch storage for datasets (no backups!)
#     /usr/bin/       — installed system binaries (python3, git, etc.)
#     /tmp/           — temporary storage, wiped at reboot
#     /proc/          — virtual filesystem: kernel exposes runtime state here
#     /dev/null       — the "black hole" device — write to discard, read EOF

echo ""
echo "=========================================================="
echo " SECTION 1: FILE SYSTEM NAVIGATION"
echo "=========================================================="

# ── 1.1 Absolute vs. Relative Navigation ─────────────────────────────────────
echo ""
echo "--- 1.1 Absolute vs. Relative Navigation ---"

# Absolute path: begins with '/', resolves from root regardless of cwd
echo "Moving with absolute path:"
cd /tmp
pwd   # should print: /tmp

# Relative path: resolves relative to cwd
echo "Moving with relative path (one level up from /tmp):"
cd ..
pwd   # should print: /  (since /tmp is a direct child of root)

# Jump home instantly with tilde shortcut
cd ~
echo "After 'cd ~', we are at: $(pwd)"
# ~ is shell-expanded to the value of $HOME before the command runs.
# This expansion happens IN THE SHELL — the 'cd' binary never sees '~'.

# Jump back to the previous directory using the $OLDPWD variable
cd /var/log
echo "Went to /var/log: $(pwd)"
cd -   # cd - is equivalent to: cd $OLDPWD
echo "After 'cd -', back to: $(pwd)"

# ── 1.2 Listing Directory Contents ───────────────────────────────────────────
echo ""
echo "--- 1.2 ls — Listing Files ---"

# Basic listing
ls /usr/bin | head -10
# head -10: pass only the first 10 lines through to stdout

# Long listing with human-readable sizes and hidden files
ls -lah /tmp
# -l : long format (permissions, owner, group, size, date, name)
# -a : all files, including hidden (dot-files like .bashrc)
# -h : human-readable sizes (1K, 23M, 4.5G instead of raw bytes)

# Sort by modification time, newest first
ls -lt /var/log | head -5
# -t : sort by modification time (newest first)

# Recursive listing — shows full tree
ls -R /tmp | head -20
# -R : recursive; WARNING: don't run on / or large trees without piping to head!

# ── 1.3 Finding Files ─────────────────────────────────────────────────────────
echo ""
echo "--- 1.3 find — Locating Files ---"

# Find all Python files starting from current directory
find . -name "*.py" -type f 2>/dev/null | head -10
# -name "*.py"  : match files ending in .py (glob pattern)
# -type f       : only regular files (not directories, symlinks)
# 2>/dev/null   : redirect stderr to /dev/null (suppress permission denied errors)

# Find files larger than 100MB (useful for spotting runaway logs)
find /tmp -size +100M -type f 2>/dev/null
# -size +100M : files strictly larger than 100 megabytes

# Find files modified in the last 30 minutes (great for checking recent outputs)
find . -mmin -30 -type f 2>/dev/null
# -mmin -30 : modified less than 30 minutes ago

# ── 1.4 Disk Usage — How Full Is Your Storage? ───────────────────────────────
echo ""
echo "--- 1.4 Disk Usage ---"

# df: filesystem-level disk free report
df -h
# -h : human-readable (shows GB/TB instead of 1K-blocks)
# Key columns: Filesystem, Size, Used, Avail, Use%, Mounted on
# On Bridges-2, watch /scratch — that's your working space

# df filtered to only show real filesystems (exclude proc/tmpfs)
df -h -x tmpfs -x devtmpfs

# du: directory-level disk usage
du -sh ~
# -s : summary (don't recurse into subdirectories — just total)
# -h : human-readable
# This tells you your total home directory size

# du on a list of directories — find what's eating your quota
du -sh ~/*/
# ~/*/  : glob expanding to all direct subdirs of $HOME

# Sort by size, largest first
du -sh /tmp/*/ 2>/dev/null | sort -rh | head -10
# sort -r : reverse order (largest first)
# sort -h : human-numeric-sort (treats 1G > 100M correctly)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: FILE PERMISSIONS — POSIX SECURITY MODEL
# ─────────────────────────────────────────────────────────────────────────────
#
# Theory:
#   Every file and directory in Linux has an owner (a user), a group, and a
#   set of permission bits controlling what owner/group/others can do.
#   The bits are: r (read=4), w (write=2), x (execute=1)
#   Octal notation combines these: rwx=7, r-x=5, rw-=6, r--=4, ---=0
#
#   Format of 'ls -l' output:
#     -rwxr-xr--  1  jsmith  ml_group  14520  Aug 28 09:14  train.py
#     ^  ^  ^  ^
#     |  |  |  └── others: r-- = 4 = read only
#     |  |  └───── group:  r-x = 5 = read + execute
#     |  └──────── owner:  rwx = 7 = full access
#     └─────────── file type: - = regular file, d = directory, l = symlink

echo ""
echo "=========================================================="
echo " SECTION 2: FILE PERMISSIONS"
echo "=========================================================="

# Create a test file to experiment with
touch /tmp/cmu_test_file.sh
echo '#!/usr/bin/env bash' > /tmp/cmu_test_file.sh
echo 'echo "Hello from test script"' >> /tmp/cmu_test_file.sh

# View initial permissions
echo "Initial permissions:"
ls -l /tmp/cmu_test_file.sh

# ── chmod: Change File Mode Bits ─────────────────────────────────────────────
echo ""
echo "--- chmod examples ---"

# Set exact permissions using octal: 755 = rwxr-xr-x
chmod 755 /tmp/cmu_test_file.sh
echo "After chmod 755:"
ls -l /tmp/cmu_test_file.sh
# Owner can read, write, execute
# Group can read and execute (but NOT write — protects code from group modification)
# Others can read and execute (world-readable scripts)

# Make a config file only owner-readable (for API keys, passwords)
touch /tmp/cmu_secret.key
chmod 600 /tmp/cmu_secret.key
echo "After chmod 600 (private key style):"
ls -l /tmp/cmu_secret.key
# rw-------: Only owner can read and write. Group and others: no access.

# Add execute bit for everyone using symbolic notation
chmod +x /tmp/cmu_test_file.sh
echo "After chmod +x (symbolic — adds execute for all):"
ls -l /tmp/cmu_test_file.sh

# Remove write bit for group and others
chmod go-w /tmp/cmu_test_file.sh
# g = group, o = others, - = remove, w = write bit

# Common permission patterns in DL workflows:
#   chmod 755 script.sh       → executable script, world-readable
#   chmod 644 config.yaml     → config file: owner rw, group+others r
#   chmod 600 ~/.ssh/id_ed25519   → private key: MUST be owner-read-only
#   chmod 700 ~/.ssh/         → SSH directory: owner-only access
#   chmod 777 /tmp/shared/    → fully open (avoid in production — security risk)

# Clean up test files
rm -f /tmp/cmu_test_file.sh /tmp/cmu_secret.key


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: PROCESS MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
#
# Theory:
#   In Linux, everything that runs is a process. Each process has:
#     PID  : unique process ID assigned by the kernel
#     PPID : parent process ID (who spawned it)
#     UID  : the user ID of the owner
#     CPU% : percentage of CPU time consumed
#     MEM% : percentage of physical RAM consumed
#     STATE: R(running), S(sleeping), D(uninterruptible), Z(zombie), T(stopped)
#
#   Processes communicate via signals:
#     SIGTERM (15): polite termination request — process can clean up
#     SIGKILL  (9): unconditional termination — kernel kills it immediately
#     SIGHUP   (1): hangup — closes when terminal closes (nohup blocks this)
#     SIGSTOP (19): suspend execution (equivalent to Ctrl+Z)

echo ""
echo "=========================================================="
echo " SECTION 3: PROCESS MANAGEMENT"
echo "=========================================================="

# ── ps: Process Snapshot ─────────────────────────────────────────────────────
echo ""
echo "--- 3.1 ps: Process Snapshot ---"

# ps aux: THE standard invocation for seeing everything
ps aux | head -15
# a : show processes for ALL users (not just yours)
# u : user-oriented format (shows USER, %CPU, %MEM columns)
# x : include processes NOT attached to a terminal (daemons, background jobs)

# Columns explained:
# USER   PID  %CPU  %MEM  VSZ    RSS   TTY  STAT  START   TIME  COMMAND
# VSZ = Virtual memory Size (total virtual address space, including swapped)
# RSS = Resident Set Size (physical RAM actually used right now)

# Find YOUR Python processes specifically
echo ""
echo "Your current python processes:"
ps aux | grep "^$(whoami)" | grep -i python || echo "(none running)"

# Find processes by name using pgrep
echo ""
echo "Process IDs matching 'bash' (pgrep):"
pgrep -l bash | head -5
# pgrep -l : list process name alongside PID
# pgrep -u $(whoami) python : find YOUR python processes

# ── Simulating and Managing a Background Process ─────────────────────────────
echo ""
echo "--- 3.2 Background Processes ---"

# Start a dummy "training simulation" in background
# sleep 120 simulates a long-running training job
sleep 120 &
BG_PID=$!
echo "Started background job with PID: $BG_PID"
# $! : special bash variable containing PID of last background command (&)

# Verify it's running
echo "Verifying background job:"
ps -p $BG_PID -o pid,stat,cmd
# -p $BG_PID : show only this specific PID
# -o pid,stat,cmd : custom output format

# List current shell's background jobs
jobs -l
# -l : show PID alongside job number

# Send it to foreground (comment this out during demo if you don't want to wait)
# fg %1     # bring job 1 to foreground

# Kill the background job
echo "Sending SIGTERM (15) — polite termination request:"
kill -15 $BG_PID
sleep 1
# Verify it's gone
if ! ps -p $BG_PID > /dev/null 2>&1; then
    echo "PID $BG_PID has been terminated (SIGTERM succeeded)"
else
    echo "Process still alive — escalating to SIGKILL (-9)"
    kill -9 $BG_PID
fi

# ── nohup: Persist Beyond Terminal Session ───────────────────────────────────
echo ""
echo "--- 3.3 nohup: Keeping Jobs Alive After Logout ---"

# nohup: No HangUP — intercepts SIGHUP signal so process survives terminal close
# The '&' sends it to background immediately
# All output (stdout + stderr) is captured to nohup.out by default

cat << 'DEMO_SCRIPT' > /tmp/cmu_fake_train.sh
#!/usr/bin/env bash
for i in $(seq 1 10); do
    echo "[$(date '+%H:%M:%S')] Epoch $i/10 — loss: $(python3 -c 'import random; print(round(random.uniform(0.1, 2.5), 4))')"
    sleep 2
done
echo "Training complete."
DEMO_SCRIPT
chmod +x /tmp/cmu_fake_train.sh

echo "Starting fake training job with nohup:"
nohup /tmp/cmu_fake_train.sh > /tmp/cmu_train_output.log 2>&1 &
TRAIN_PID=$!
echo "nohup job started. PID: $TRAIN_PID"
echo "Output redirected to: /tmp/cmu_train_output.log"

# Wait a moment, then show live output
sleep 3
echo ""
echo "Live log output (tail -f style):"
tail -5 /tmp/cmu_train_output.log

# Kill the training job when done demonstrating
kill $TRAIN_PID 2>/dev/null
wait $TRAIN_PID 2>/dev/null
echo "Demo training job terminated."

# ── tmux Primer ──────────────────────────────────────────────────────────────
echo ""
echo "--- 3.4 tmux: Terminal Multiplexer Reference ---"
cat << 'TMUX_REF'

  tmux new -s training         # Create a new session named "training"
  tmux attach -t training      # Re-attach to existing "training" session
  tmux ls                      # List all active tmux sessions
  tmux kill-session -t name    # Kill a specific session

  INSIDE tmux — Key Bindings (prefix is Ctrl+B):
    Ctrl+B, D           → Detach (leave session running in background)
    Ctrl+B, C           → Create new window
    Ctrl+B, N/P         → Next/Previous window
    Ctrl+B, %           → Split pane vertically
    Ctrl+B, "           → Split pane horizontally
    Ctrl+B, Arrow Keys  → Navigate between panes
    Ctrl+B, [           → Enter scroll mode (use arrow keys, q to exit)

  USE CASE: SSH into Bridges-2 → tmux new -s dl_run → start training →
            Ctrl+B D → close laptop → reconnect next day →
            ssh bridges2 → tmux attach -t dl_run → job still running!

TMUX_REF

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: GPU MONITORING WITH nvidia-smi
# ─────────────────────────────────────────────────────────────────────────────
#
# Theory:
#   nvidia-smi (System Management Interface) communicates with the NVIDIA
#   Management Library (NVML) which talks directly to the GPU driver.
#   It reports: GPU utilization, VRAM used/free, temperature, power draw,
#   running compute processes, PCIe bandwidth, and clock speeds.
#
#   VRAM (Video RAM) is the GPU's dedicated on-device memory. In DL:
#     - Model parameters live in VRAM
#     - Batch tensors live in VRAM during forward/backward pass
#     - Gradients live in VRAM during backprop
#   If VRAM is exhausted: CUDA out of memory (OOM) → training crashes.

echo ""
echo "=========================================================="
echo " SECTION 4: GPU MONITORING"
echo "=========================================================="

echo ""
echo "--- 4.1 nvidia-smi: Full Status Report ---"

if command -v nvidia-smi &>/dev/null; then
    # Basic one-shot status
    nvidia-smi

    echo ""
    echo "--- 4.2 Filtered VRAM query (CSV format) ---"
    nvidia-smi \
        --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits
    # --query-gpu : comma-separated list of metrics to pull
    # --format=csv : machine-readable CSV output
    # noheader : suppress column header row
    # nounits  : strip MiB/% labels (useful for scripted parsing)

    echo ""
    echo "--- 4.3 Continuous monitoring (press Ctrl+C to stop after 3 iterations) ---"
    # nvidia-smi dmon: device monitor — prints tabular stats at interval
    # -s u : show utilization stats (compute + memory utilization %)
    # -d 1 : refresh every 1 second
    # -c 3 : capture only 3 iterations (remove for continuous monitoring)
    nvidia-smi dmon -s u -d 1 -c 3

    echo ""
    echo "--- 4.4 Per-process GPU usage ---"
    # nvidia-smi pmon: process monitor — shows GPU usage per running process
    nvidia-smi pmon -c 1
    # Shows: PID, process name, GPU memory used by that process

    echo ""
    echo "--- 4.5 Scripted VRAM check with threshold alert ---"
    VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    echo "VRAM Used: ${VRAM_USED} MiB"
    echo "VRAM Free: ${VRAM_FREE} MiB"

    if [ "$VRAM_FREE" -lt 2048 ]; then
        echo "WARNING: Less than 2GB VRAM free — reduce batch size or free allocations!"
    else
        echo "OK: Sufficient VRAM available for typical DL workloads."
    fi

else
    echo "nvidia-smi not available on this machine."
    echo "On a GPU node, this command would show GPU status."
    echo "Tip: On Bridges-2, run 'srun --gres=gpu:1 --pty bash' first."
fi


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: TEXT STREAM PROCESSING — grep, awk, sed, PIPES
# ─────────────────────────────────────────────────────────────────────────────
#
# Theory:
#   The Unix philosophy: programs that do ONE thing well and compose through
#   pipes. The pipe operator '|' takes the STDOUT of the left command and
#   connects it directly to STDIN of the right command — all in-memory,
#   no temp files required. This enables powerful data pipelines.
#
#   grep: filters lines matching a pattern (regex or literal)
#   awk:  column-based processing — treats each line as fields ($1, $2, ...)
#   sed:  stream editor — transforms text via s/find/replace/ substitutions
#   sort: sorts stdin lines (lexicographic, numeric, human-numeric)
#   uniq: de-duplicates consecutive identical lines (sort first!)
#   wc:   counts lines (-l), words (-w), bytes (-c)
#   head/tail: first/last N lines of stdin

echo ""
echo "=========================================================="
echo " SECTION 5: TEXT STREAM PROCESSING"
echo "=========================================================="

# ── Create a simulated training log file for our exercises ───────────────────
cat > /tmp/cmu_training.log << 'LOG_EOF'
[2026-08-28 09:00:01] INFO  Starting training — model: ResNet50, dataset: ImageNet
[2026-08-28 09:00:02] DEBUG DataLoader initialized — workers: 8, batch_size: 256
[2026-08-28 09:00:03] DEBUG Pinning memory to GPU:0
[2026-08-28 09:01:45] INFO  Epoch 1/50 — train_loss: 2.3451, train_acc: 0.1823
[2026-08-28 09:01:45] INFO  Epoch 1/50 — val_loss: 2.4102, val_acc: 0.1742
[2026-08-28 09:03:22] INFO  Epoch 2/50 — train_loss: 2.1837, train_acc: 0.2341
[2026-08-28 09:03:22] INFO  Epoch 2/50 — val_loss: 2.2090, val_acc: 0.2284
[2026-08-28 09:05:01] DEBUG Checkpoint saved: /scratch/jsmith/checkpoints/epoch_2.pt
[2026-08-28 09:06:38] INFO  Epoch 3/50 — train_loss: 1.9924, train_acc: 0.3015
[2026-08-28 09:06:38] INFO  Epoch 3/50 — val_loss: 2.0548, val_acc: 0.2891
[2026-08-28 09:06:38] WARNING VRAM usage at 94% — consider reducing batch size
[2026-08-28 09:08:15] INFO  Epoch 4/50 — train_loss: 1.8201, train_acc: 0.3602
[2026-08-28 09:08:15] INFO  Epoch 4/50 — val_loss: 1.8899, val_acc: 0.3488
[2026-08-28 09:09:52] INFO  Epoch 5/50 — train_loss: 1.6790, train_acc: 0.4105
[2026-08-28 09:09:52] INFO  Epoch 5/50 — val_loss: 1.7423, val_acc: 0.3994
[2026-08-28 09:09:52] DEBUG Checkpoint saved: /scratch/jsmith/checkpoints/epoch_5.pt
[2026-08-28 09:09:53] ERROR CUDA error: device-side assert triggered on GPU:0
LOG_EOF

echo "Created simulated training log: /tmp/cmu_training.log"
echo ""

# ── 5.1 grep — Pattern Matching ───────────────────────────────────────────────
echo "--- 5.1 grep: Pattern Filtering ---"

echo "Lines containing 'Epoch':"
grep "Epoch" /tmp/cmu_training.log

echo ""
echo "Lines containing 'Epoch' with line numbers (-n):"
grep -n "Epoch" /tmp/cmu_training.log

echo ""
echo "Lines NOT containing 'DEBUG' (-v inverts match):"
grep -v "DEBUG" /tmp/cmu_training.log

echo ""
echo "Only ERROR or WARNING lines (extended regex -E, alternation |):"
grep -E "ERROR|WARNING" /tmp/cmu_training.log

echo ""
echo "Case-insensitive search (-i) for 'cuda':"
grep -i "cuda" /tmp/cmu_training.log

echo ""
echo "Count how many INFO lines exist (-c returns count, not lines):"
grep -c "INFO" /tmp/cmu_training.log

# ── 5.2 awk — Field-Based Processing ─────────────────────────────────────────
echo ""
echo "--- 5.2 awk: Column Extraction & Logic ---"

# awk splits each input line into fields by whitespace (default delimiter)
# $1=first field, $2=second, ..., $NF=last field, NF=number of fields

echo "Extract just the timestamp from each log line (\$1 = date, \$2 = time):"
awk '{print $1, $2}' /tmp/cmu_training.log

echo ""
echo "Print only training loss values (lines with 'train_loss', get last field):"
grep "train_loss" /tmp/cmu_training.log | awk '{print "Epoch:", $5, "Loss:", $NF}'
# $5 is the epoch field (e.g. "1/50"), $NF is always the last field

echo ""
echo "awk conditional: only print lines where val_loss < 2.0 (requires numeric comparison):"
grep "val_loss" /tmp/cmu_training.log | awk '{
    # Split the last field "val_acc:X.XXXX" on ":" to get the number
    split($NF, a, ":");
    if (a[2] > 0.35) print "HIGH ACCURACY:", $0
}'

echo ""
echo "awk to sum all training losses and compute mean:"
grep "train_loss" /tmp/cmu_training.log | awk -F': ' '{
    split($NF, parts, ",");
    loss = parts[1];
    total += loss;
    count++;
}
END { printf "Mean training loss: %.4f over %d epochs\n", total/count, count }'
# -F': ' : change the field separator to ': ' (colon-space)
# END block: executes after all input lines are processed

# ── 5.3 sed — Stream Editing ──────────────────────────────────────────────────
echo ""
echo "--- 5.3 sed: Text Transformation ---"

echo "Replace 'DEBUG' with '[DBG]' in the log stream (not in-file):"
sed 's/DEBUG/[DBG]/g' /tmp/cmu_training.log | head -5
# s/pattern/replacement/g
# s = substitute, g = global (replace ALL occurrences per line, not just first)

echo ""
echo "Delete all DEBUG lines from output (-n suppresses, /p prints matches):"
sed '/DEBUG/d' /tmp/cmu_training.log

echo ""
echo "Print only lines 4 through 8 (line number addressing):"
sed -n '4,8p' /tmp/cmu_training.log
# -n : suppress all output unless explicitly printed
# '4,8p' : print lines 4 through 8

echo ""
echo "Replace multiple config values in a file (in-place with -i):"
# Make a test config
cat > /tmp/cmu_test_config.yaml << 'CONF'
learning_rate: 0.01
batch_size: 256
num_epochs: 50
dropout: 0.5
CONF

echo "Before:"
cat /tmp/cmu_test_config.yaml

# In-place replacement: -i makes changes to the actual file
sed -i 's/learning_rate: 0\.01/learning_rate: 0.001/' /tmp/cmu_test_config.yaml
sed -i 's/batch_size: 256/batch_size: 128/' /tmp/cmu_test_config.yaml

echo "After (learning_rate and batch_size changed):"
cat /tmp/cmu_test_config.yaml

# ── 5.4 Chained Pipelines ─────────────────────────────────────────────────────
echo ""
echo "--- 5.4 Pipe Chains: Composing Powerful Workflows ---"

echo "Full pipeline: grep → awk → sort → tail"
echo "Goal: Extract val_acc values, sort ascending, show top 3 (best epochs):"
grep "val_acc" /tmp/cmu_training.log \
    | awk '{split($NF,a,":"); print a[2], $0}' \
    | sort -k1 -rn \
    | head -3
# grep: filter to val_acc lines only
# awk: extract the val_acc number as first field, keep original line
# sort -k1 -rn: sort on field 1 (-k1), reverse (-r), numerically (-n)
# head -3: show only top 3 results

echo ""
echo "Monitor top 5 memory-consuming processes right now:"
ps aux --sort=-%mem | head -6
# --sort=-%mem : sort by %MEM descending (- prefix = descending)

echo ""
echo "Count unique error types in the log:"
grep -E "ERROR|WARNING" /tmp/cmu_training.log \
    | awk '{print $3}' \
    | sort \
    | uniq -c \
    | sort -rn
# sort | uniq -c : count occurrences of each unique line
# final sort -rn : show most frequent first


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: WORKING WITH ARCHIVES & COMPRESSED DATA
# ─────────────────────────────────────────────────────────────────────────────
#
# Theory:
#   tar (Tape ARchive) is the standard Unix archiving tool. By itself, tar
#   bundles files — it does NOT compress. Compression is added by piping
#   through gzip (-z flag), bzip2 (-j), or xz (-J).
#
#   gzip uses the DEFLATE algorithm (LZ77 + Huffman coding). It offers good
#   speed with moderate compression. For ML datasets with structured binary
#   data (images, numpy arrays), compression ratios vary:
#     - Raw text configs/logs: 60-80% size reduction
#     - JPEG images: <5% reduction (already compressed)
#     - Numpy .npy arrays: 30-70% depending on sparsity
#   For sparse matrices and embeddings, consider lz4 (faster) or zstd (better ratio).

echo ""
echo "=========================================================="
echo " SECTION 6: ARCHIVES AND COMPRESSION"
echo "=========================================================="

# Create a mock dataset directory structure
mkdir -p /tmp/cmu_dataset/{train,val,test}
for split in train val test; do
    for i in $(seq 1 5); do
        dd if=/dev/urandom of="/tmp/cmu_dataset/${split}/sample_${i}.bin" \
           bs=1024 count=64 2>/dev/null
        # Creates 64KB random binary files (simulating tensors)
    done
done
echo "Created mock dataset in /tmp/cmu_dataset/"
echo "Size before compression:"
du -sh /tmp/cmu_dataset/

echo ""
echo "--- 6.1 Creating tar.gz Archive ---"
# -c : create new archive
# -z : filter through gzip compression
# -f : next argument is the archive filename (MUST come last before filename)
# -v : verbose (list each file as it's added)
tar -czf /tmp/cmu_dataset_v1.tar.gz /tmp/cmu_dataset/ 2>/dev/null
echo "Archive created."
echo "Compressed archive size:"
du -sh /tmp/cmu_dataset_v1.tar.gz

echo ""
echo "--- 6.2 Inspect Archive Contents (without extracting) ---"
tar -tzf /tmp/cmu_dataset_v1.tar.gz | head -15
# -t : list contents (table of contents)
# -z : decompress through gzip first
# -f : archive filename

echo ""
echo "--- 6.3 Extract to Specific Directory ---"
mkdir -p /tmp/cmu_extracted/
tar -xzf /tmp/cmu_dataset_v1.tar.gz -C /tmp/cmu_extracted/
# -x : extract files
# -z : decompress through gzip
# -f : archive filename
# -C : change to this directory before extracting (crucial for path control)
echo "Extracted to /tmp/cmu_extracted/"
ls /tmp/cmu_extracted/tmp/cmu_dataset/

echo ""
echo "--- 6.4 Extract Only Specific Files ---"
# Extract only the 'train' directory from the archive
tar -xzf /tmp/cmu_dataset_v1.tar.gz -C /tmp/ \
    --strip-components=3 \
    "tmp/cmu_dataset/train/"
# --strip-components=3 : remove the first 3 path components when extracting
# so "tmp/cmu_dataset/train/sample_1.bin" → "sample_1.bin" in /tmp/

# Cleanup
rm -rf /tmp/cmu_dataset/ /tmp/cmu_extracted/ /tmp/cmu_dataset_v1.tar.gz
rm -f /tmp/cmu_training.log /tmp/cmu_train_output.log /tmp/cmu_fake_train.sh
rm -f /tmp/cmu_test_config.yaml


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: WORKSPACE SETUP FOR DEEP LEARNING
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "=========================================================="
echo " SECTION 7: SETTING UP A DL PROJECT WORKSPACE"
echo "=========================================================="

echo "Creating standard DL project directory scaffold:"

# The full directory scaffold a well-organized DL project needs
DL_PROJECT="${HOME}/demo_dl_project"

mkdir -p "${DL_PROJECT}"/{data/{raw,processed,external},\
models/{architectures,pretrained},\
checkpoints,\
logs/{training,validation,tensorboard},\
results/{figures,tables},\
scripts/{preprocessing,training,evaluation},\
configs,\
notebooks}

echo "Directory tree created at: ${DL_PROJECT}"
find "${DL_PROJECT}" -type d | sort | sed 's|[^/]*/|  |g'

echo ""
echo "Creating placeholder files for project structure:"

# .gitignore — essential for not committing large data files
cat > "${DL_PROJECT}/.gitignore" << 'GITIGNORE'
# Data files — never commit large datasets to git
data/raw/
data/processed/
*.tar.gz
*.zip
*.npy
*.pt
*.ckpt

# Checkpoints
checkpoints/

# Logs and results
logs/
results/

# Python
__pycache__/
*.pyc
.venv/

# Jupyter
.ipynb_checkpoints/
GITIGNORE

# README skeleton
cat > "${DL_PROJECT}/README.md" << 'README'
# Deep Learning Project

## Project Structure
- `data/`       — raw and processed dataset storage
- `models/`     — model architecture definitions
- `checkpoints/`— saved training states
- `logs/`       — training and validation metrics
- `scripts/`    — preprocessing, training, evaluation
- `configs/`    — hyperparameter and experiment configs
- `notebooks/`  — exploratory analysis

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
README

echo "Created .gitignore and README.md"

echo ""
echo "--- Checking Key Dependency Availability ---"

# Python
if command -v python3 &>/dev/null; then
    PYTHON_VER=$(python3 --version 2>&1)
    echo "✓ Python: $PYTHON_VER"
    PYTHON_PATH=$(which python3)
    echo "  Location: $PYTHON_PATH"
else
    echo "✗ python3: NOT FOUND — install via conda or system package manager"
fi

# pip
if command -v pip3 &>/dev/null; then
    echo "✓ pip3: $(pip3 --version 2>&1 | awk '{print $1, $2}')"
else
    echo "✗ pip3: NOT FOUND"
fi

# PyTorch
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
    CUDA_AVAIL=$(python3 -c "import torch; print(torch.cuda.is_available())")
    CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)")
    echo "✓ PyTorch: version $TORCH_VER"
    echo "  CUDA available: $CUDA_AVAIL"
    echo "  CUDA version:   $CUDA_VER"
else
    echo "✗ PyTorch: NOT INSTALLED — run: pip install torch torchvision"
fi

# CUDA toolkit
if command -v nvcc &>/dev/null; then
    echo "✓ CUDA Compiler: $(nvcc --version | grep 'release' | awk '{print $5,$6}')"
else
    echo "  nvcc: not in PATH (normal on login nodes; available on GPU nodes)"
fi

# git
if command -v git &>/dev/null; then
    echo "✓ git: $(git --version)"
fi

echo ""
echo "Workspace setup complete: ${DL_PROJECT}"

# Optional cleanup
# rm -rf "${DL_PROJECT}"


# ─────────────────────────────────────────────────────────────────────────────
# END OF MASTERCLASS
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "=========================================================="
echo " CMU 11-785 Linux CLI Masterclass — Complete"
echo ""
echo " Review each section, re-run commands individually,"
echo " and experiment with your own variations."
echo " The best way to learn Linux is to break things and"
echo " figure out how to fix them."
echo "=========================================================="
