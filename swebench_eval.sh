

# condenser=summarizer_for_eval
# condenser=summarizer_for_eval_2507_32768
# condenser=summarizer_for_eval_2507_32768_max30
# condenser=recent_for_eval_max60
condenser=
# condenser=summarizer_for_eval_2507_65536_FP8_max60
# condenser=recent_for_eval
# model="llm.devstral-128k"
# model="llm.devstral-64k"
# model="llm.devstral-2507-90k"
# model="llm.devstral-2507-32768"
# model="llm.devstral-2507-65536-FP8"
model="llm.devstral-2507-w8-131072-FP8"
# model="llm.devstral-2507-131072-FP8"
# model="llm.devstral-2507-65536-FP8"
# model="llm.devstral-32768"
# model="llm.devstral-65536"
# model="llm.deepseek-chat"
maxiter=100
N=1
dataset="MariusHobbhahn/swe-bench-verified-mini"
agent="CodeActAgent"
eval_limit=50


export ITERATIVE_EVAL_MODE=true

log_name="${model}_${condenser}_${dataset}_${agent}_${eval_limit}_${maxiter}_${N}"
if [ "$ITERATIVE_EVAL_MODE" = true ]; then
    log_name="${log_name}-iter"
fi

export EVAL_CONDENSER=${condenser}
export USE_HINT_TEXT=false
export ENABLE_LLM_EDITOR=false
export DEBUG=1

echo "" > ./logs/openhands.log
./evaluation/benchmarks/swe_bench/scripts/run_infer.sh ${model} HEAD ${agent} ${eval_limit} ${maxiter} ${N} ${dataset} test
log_name=${log_name//\//-}
mv ./logs/openhands.log ./logs/${log_name}.log

# exit

if [ -z "$condenser" ]; then
    outdir=./evaluation/evaluation_outputs/outputs/${dataset//\//__}-test/${agent}/${model//llm./}_maxiter_${maxiter}_N_v0.56.0-no-hint-run_${N}
else
    outdir=./evaluation/evaluation_outputs/outputs/${dataset//\//__}-test/${agent}/${model//llm./}_maxiter_${maxiter}_N_v0.56.0-no-hint-${condenser}-run_${N}
fi


if [ "$ITERATIVE_EVAL_MODE" = true ]; then
    outdir=${outdir}-iter
fi

./evaluation/benchmarks/swe_bench/scripts/eval_infer.sh \
    ${outdir}/output.jsonl \
    "" ${dataset}
