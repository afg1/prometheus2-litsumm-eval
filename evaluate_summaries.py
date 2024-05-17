from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT_WO_REF, SCORE_RUBRIC_TEMPLATE

from datasets import load_dataset
import polars as pl
import re

pmcid_pattern = re.compile(r"PMC\d+")


judge = PrometheusEval(model_id="prometheus-eval/prometheus-7b-v2.0", absolute_grade_template=ABSOLUTE_PROMPT_WO_REF)

litsumm_instruction = (
        "As an experienced academic who ALWAYS provides references for each sentence you write, "
        "produce a summary from the text below, focusing on {ent_id} and using the references for each sentence. "
        "\n\n{context}\n\n"
        "The reference for each sentence in the text is given at the end of the sentence, enclosed by []. "
        "For example, the first sentence has the reference [{first_ref}]. "
        "Refrences should only be provided at the end of sentences, and MUST follow the style in the context. Do not list references at the end of the summary. "
        "You MUST provide at least one reference per sentence you produce. "
        "Use only the information in the context given above. Start your summary with a brief description of {ent_id}, noting its type. "
        "Use 200 words or less."
        "\nSummary:\n"
    )

rubric_data = {
    "criteria":"",
    "score1_description": "Serious problems in the summary, for example:\n-Incorrect reference format, total lack of references, restatement of references at the end of the summary\n-Hallucinated references, e.g. in author-year style\n-Multiple egregiously incorrect statements\n-Any mention of being an AI/LLM",
    "score2_description": "A problematic summary with two incorrect/misleading statements, or one egregiously wrong statement. Large parts of the important information in the context are missing from the summary. References are given in the correct format, but may be inadequate. Style/flow may be poor.",
    "score3_description": "An acceptable summary. Style or flow may be of lower quality than a human might write. May miss considerable amounts of information. At most one incorrect or misleading statement. All references are real and given in the correct format.",
    "score4_description": "A high quality summary, but with some small issues, for example style, flow, construction or missing information. All statements are factual and supported by the information in the context, references are real, adequate and presented correctly. Could be rescued by a few small changes",
    "score5_description": "Excellent quality summary, as good or nearly as good as one written by a human curator. Fully referenced in the correct format, with all statements supported by the context."
}

score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

def prepare_evaluation(row):
    summary = row['summary']
    context = row['context']
    ent_id = row['ent_id']
    first_ref = pmcid_pattern.findall(context)[0]

    instruction = litsumm_instruction.format(ent_id=ent_id, context=context, first_ref=first_ref)

    return {"instruction": instruction, "response": summary}





## Load data
dataset = load_dataset("RNAcentral/litsumm-v1.5")['train']
dataframe = pl.from_arrow(dataset.data.table)
print(dataframe)


dataframe = dataframe.with_columns(res=pl.struct(pl.col("context"), pl.col("summary"), pl.col("ent_id")).map_elements(prepare_evaluation)).unnest("res")

instructions = dataframe.get_column("instruction").to_list()
responses = dataframe.get_column("response").to_list()

feedbacks, scores = judge.absolute_grade(
    instructions=instructions,
    responses=responses,
    rubric=score_rubric
)

dataframe = dataframe.with_columns(feedback=pl.Series(feedbacks))
dataframe = dataframe.with_columns(scores=pl.Series(scores))

dataframe.write_parquet("litsumm_v1.5_rated_prometheus2.parquet")
