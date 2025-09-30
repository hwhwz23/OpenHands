
attempts_dir=./evaluation/evaluation_outputs/outputs/MariusHobbhahn__swe-bench-verified-mini-test/CodeActAgent/devstral-2507-65536-FP8_maxiter_100_N_v0.56.0-no-hint-run_1-iter

dataset="MariusHobbhahn/swe-bench-verified-mini"


# list all files matching output.critic_attempt_*.jsonl
attempts_files=$(ls ${attempts_dir}/output.critic_attempt_*.jsonl)

for attempt_file in ${attempts_files}; do
    base_name=$(basename ${attempt_file} .jsonl)
    echo "processing... ${base_name}"
    new_dir=${attempts_dir}/${base_name}
    mkdir -p ${new_dir}
    cp ${attempt_file} ${new_dir}/output.jsonl
    ./evaluation/benchmarks/swe_bench/scripts/eval_infer.sh \
        ${new_dir}/output.jsonl \
        "" ${dataset}
done
