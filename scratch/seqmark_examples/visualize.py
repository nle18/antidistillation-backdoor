import sys
import nltk
import torch

from utils.io_utils import *


def highlight_and_pretty_print(output, i):

    # output example id, prompt, and info
    html_content = f"<h4>Example ID={i}</h4>"
    html_content += f"<p><b>German</b>: {output['German']}</p>"
    for answer_type, answers in output.items():

        if answer_type in ["English"]:
            for idx, answer in enumerate(answers):  # answer is a dict

                if "postprocessed" in answer and answer["postprocessed"]:
                    # green/red token mastk
                    if answer_type in ["answers (non-watermark)", "answers (watermark)"]:
                        masks = answer["watermark_score"]["green_token_mask"]
                    else:
                        masks = answer["tokenized_text"]
                else:

                    # get rid of start tokens (just a bug when generating-- I use convert_ids_to_tokens, which adds extra <s>)
                    if answer_type == "English":
                        answer["tokenized_text"] = answer["tokenized_text"][1:]
                        answer["token_entropies"] = answer["token_entropies"][1:]
                        answer["top-10"] = answer["top-10"][1:]
                        # green/red token mastk
                        masks = answer["tokenized_text"]
                    else:
                        masks = [None, None, None] + answer["watermark_score"]["green_token_mask"]

                # then we output answer with highlighting
                # first normalize the entropy to get an alpha value between 0 and 1
                answer["entropy_alphas"] = [e / max(answer["token_entropies"]) for e in answer["token_entropies"]]

                # construct content based on highlighting the entropy
                content = ""
                for i, (tok, alpha, mask) in enumerate(list(zip(answer["tokenized_text"], answer["entropy_alphas"], masks))):
                    color = f"rgba(255, 176, 66, {alpha})"  # yellow-orange
                    # manually replace the special '_' tokens with space for human-easy reading
                    tok = tok.replace("▁", " ").replace("<0x0A>", " ")

                    # generate top-10 entropy content
                    topk_content = f"entropy: {answer['token_entropies'][i]:0.4f}\n"  # first add the entropy value
                    topk_content += "-" * 10 + "\n"
                    for candidate, prob, cand_color in answer["top-10"][i]:
                        candidate = candidate.replace("▁", " ").replace("<0x0A>", " ").replace('"', "'")
                        topk_content += f"{candidate}: {prob:0.4f}\n"

                    if mask is True:
                        token_color = "green"
                    elif mask is False:
                        token_color = "red"
                    else:
                        token_color = "black"
                    content += f'<span style=\'background-color: {color}; color: {token_color}\' \
class="hoverable"\
data-hover-content="{topk_content}">\
{tok}\
</span>'
                # pretty print
                html_content += f"<p><b>{answer_type}</b>: {content}"

                # add scoring info
                for key, value in answer.items():

                    # entropy
                    if (key == "avg_entropy_no_first" and answer_type in ["answers (reference)"]) or (
                        key == "avg_entropy" and answer_type in ["answers (non-watermark)", "answers (watermark)"]
                    ):
                        html_content += f"<b>{key}</b>: {value:0.4f}, "

                    # watermark metrics
                    if key in ["watermark_score"] and answer_type in ["answers (non-watermark)", "answers (watermark)"]:
                        html_content += f"<b>z-score</b>: {value['z_score']:0.4f}, <b>p-value</b>: {value['p_value']:0.4f}, <b>num_tokens_scored</b>: {value['num_tokens_scored']}, <b>num_green_tokens</b>: {value['num_green_tokens']}, <b>green_fraction</b>: {100*value['green_fraction']:0.2f}% | "
                    elif key in ["mt_score"] and answer_type in ["answers (non-watermark)", "answers (watermark)"]:
                        html_content += f"<b>BLEU</b>: {value['bleu']:0.4f}"

                html_content += "</p>"
    return html_content


def main():

    input_filepath = "./scratch/seqmark_examples/translation.jsonl"
    output_filepath = "./scratch/seqmark_examples/translation.html"
    outputs = read_jsonl(input_filepath)
    print(len(outputs))
    print(outputs[0].keys())
    html_content = "<html><head><title>Exp 3 WMT19 Entropy</title></head><body>"
    # create hover text
    # content: "First line.\ASecond line.";
    # content: attr(data-hover-content);
    html_content += """\
<style>
    .hoverable {
        position: relative;
        cursor: pointer;
    }
    .hoverable:hover::after {
        content: attr(data-hover-content);
        white-space: pre-line;
        position: absolute;
        top: 100%;
        left: 0;
        color: black;
        background-color: #FFFFFF;
        padding: 5px;
        border: 2px solid #ccc;
        border-radius: 5px;
        z-index: 100;
        width: 180px; 
    }
</style>"""
    for i, output in enumerate(outputs):
        content = highlight_and_pretty_print(output, i)
        html_content += content
    write_txt([html_content], output_filepath)


if __name__ == "__main__":
    main()
