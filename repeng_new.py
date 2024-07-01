from functools import reduce
import re
import json
import tqdm
import torch
​
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel, DatasetEntry
# import from file.
#from config import BIG5
BIG5 = {
    'EXT': [
        ('unfriendly', 'friendly'),
        ('introverted', 'extraverted'),
        ('silent', 'talkative'),
        ('timid', 'bold'),
        ('unassertive', 'assertive'),
        ('inactive', 'active'),
        ('unenergetic', 'energetic'),
        ('unadventurous', 'adventurous and daring'),
        ('gloomy', 'cheerful')
    ],
    'AGR': [
        ('distrustful', 'trustful'),
        ('immoral', 'moral'),
        ('dishonest', 'honest'),
        ('unkind', 'kind'),
        ('stingy', 'generous'),
        ('unaltruistic', 'altruistic'),
        ('uncooperative', 'cooperative'),
        ('self-important', 'humble'),
        ('unsympathetic', 'sympathetic'),
        ('selfish', 'unselfish'),
        ('disagreeable', 'agreeable')
    ],
    'CON': [
        ('unsure', 'self-efficacious'),
        ('messy', 'orderly'),
        ('irresponsible', 'responsible'),
        ('lazy', 'hardworking'),
        ('undisciplined', 'self-disciplined'),
        ('impractical', 'practical'),
        ('extravagant', 'thrifty'),
        ('disorganized', 'organized'),
        ('negligent', 'conscientious'),
        ('careless', 'thorough')
    ],
    'NEU': [
        ('relaxed', 'tense'),
        ('at ease', 'nervous'),
        ('easygoing', 'anxious'),
        ('calm', 'angry'),
        ('patient', 'irritable'),
        ('happy', 'depressed'),
        ('unselfconscious', 'self-conscious'),
        ('level-headed', 'impulsive'),
        ('contented', 'discontented'),
        ('emotionally stable', 'emotionally unstable')
    ],
    'OPE': [
        ('unimaginative', 'imaginative'),
        ('uncreative', 'creative'),
        ('artistically unappreciative', 'artistically appreciative'),
        ('unaesthetic', 'aesthetic'),
        ('unreflective', 'reflective'),
        ('emotionally closed', 'emotionally aware'),
        ('uninquisitive', 'curious'),
        ('predictable', 'spontaneous'),
        ('unintelligent', 'intelligent'),
        ('unanalytical', 'analytical'),
        ('unsophisticated', 'sophisticated'),
        ('socially conservative', 'socially progressive')
    ],
}
​
user_tag, asst_tag = "[INST]", "[/INST]"
​
# !!!CHANGE THIS!!! basic settings.
HUGGING_FACE_TOKEN = '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'
login(token=HUGGING_FACE_TOKEN)
​
​
class make_control_vector:
    def __init__(self, positive_personas, negative_personas, dataset_num):
        self.positive_personas = positive_personas
        self.negative_personas = negative_personas
        self.dataset_num = dataset_num
​
    def forward(self):
        with open("all_truncated_outputs.json") as f:
            suffixes = json.load(f)
​
        def template(persona: str, suffix: str) -> str:
            return f"{user_tag} Pretend you're an {persona} person making statements about the world. {asst_tag} {suffix}"
​
        dataset = []
        for suffix in suffixes:
            tokens = tokenizer.tokenize(suffix)
            # we augment our short suffix list by taking lots of different truncations.
            # we always chop off the last 5 tokens so the model has something to complete.
            for i in range(1, len(tokens)-self.dataset_num):
                truncated = tokenizer.convert_tokens_to_string(tokens[:i])
                for positive_persona, negative_persona in zip(self.positive_personas, self.negative_personas):
                    dataset.append(
                        DatasetEntry(
                            positive=template(positive_persona, truncated),
                            negative=template(negative_persona, truncated),
                        )
                    )
        model.reset()  # make sure you always reset the model before training a new vector
        control_vector = ControlVector.train(
            model,
            tokenizer,
            dataset,
        )
        return control_vector
​
​
def load_questions(questionnaire_fn):
    def _convert_statement(text):
        if not text.startswith('I '):
            text = f'I {text[0].lower()}{text[1:]}'
        if not text.endswith('.'):
            text = text + '.'
        return text
​
    with open(questionnaire_fn, 'r') as F:
        data = json.load(F)
​
    for d in data:
        d['statement'] = _convert_statement(d['statement'])
​
    return data
​
​
def create_personality_vector(positive_list, negative_list, dataset_num=2, method='combine_vector'):
    assert method in ['combine_dataset', 'combine_vector']
    if method == 'combine_dataset':
        result = make_control_vector(
            positive_list, negative_list, dataset_num=dataset_num)
        control_vector = result.forward()
        return control_vector
​
    # another posibility?
    elif method == 'combine_vector':
        ctrl_vectors = []
        for neg_adj, pos_adj in zip(positive_list, negative_list):
            result = make_control_vector(
                [pos_adj], [neg_adj], dataset_num=dataset_num)
            vec = result.forward()
            ctrl_vectors.append(vec)
​
        # add the vectors.
        control_vector = reduce(lambda x, y: x + y, ctrl_vectors)
        # scale the control vector.
        control_vector = (1/len(ctrl_vectors)) * control_vector
​
​
def personality_test(questions, model, vector, scalar, settings, target_dim=None):
    model.reset()
    model.set_control(vector, scalar)
​
    answers = {}
    for q_num, question in tqdm.tqdm(enumerate(questions)):
        # skip the question.
        if target_dim is not None and target_dim != question['dimension']:
            continue
​
        # prompt.
        statement = question['statement']
        prompt = f"{user_tag} Act as a person and evaluate the statement, {statement}. "
        prompt += 'Please rate how accurately this describes you on a scale from 1 to 5. (where 1="very inaccurate", 2="moderately inaccurate", 3="neither accurate nor inaccurate", 4="moderately accurate", and 5="very accurate").'
        prompt += f'Please answer using EXACTLY one of the following: 1, 2, 3, 4, or 5. {asst_tag}'
        prompt += ' I would rate this statement as '
​
        # get response.
        input_ids = tokenizer(
            prompt, return_tensors="pt").to(model.device)
​
        # try one more time if choice is None.
        choice = None
        for _ in range(2):
            res = tokenizer.decode(model.generate(
                **input_ids, **settings).squeeze())
​
            #choice = extract_and_convert(res)
            res = res.split(asst_tag)[1]
            choice = _find_answer_from_string(res)
            if choice is not None:
                break
​
        answers[q_num] = {'statement': question['statement'], "dimension": question['dimension'],
                          "math": question['math'], "answer": res, "choice": choice}
​
        # print(answers[q_num])
​
    return answers
​
​
def _find_answer_from_string(s):
    if s in ['1', '2', '3', '4', '5', 1, 2, 3, 4, 5]:
        return int(s)
​
    # Define the pattern for special characters
    pattern = re.compile(r'[12345]')
    # Search for the first special character
    match = pattern.search(s)
    # Return the special character if found, otherwise return None
​
    return int(match.group()) if match else None
​
​
def extract_and_convert(text):
    # 使用正则表达式提取[/INST]后面的数字
    match = re.search(r'\[/INST\]\s*(\d+)', text)
    if match:
        # 将匹配的数字字符串转换为整数
        return int(match.group(1))
    else:
        # 如果没有找到匹配，返回None或者其他适当的值
        print("no")
        return None
​
​
def score_report(answers):
    dimension_wise_scores = {dim: [] for dim in BIG5}
    for _, ans in answers.items():
        dim = ans['dimension']
​
        # convert scores properly.
        #score = extract_and_convert(ans['answer'])
        score = _find_answer_from_string(ans['answer'])
​
        if score in [1, 2, 3, 4, 5]:
            # reverse the scores according to the math sign.
            if dim != 'NEU' and ans['math'] == '-':
                score = 5 - (score - 1)
            elif dim == 'NEU' and ans['math'] == '+':
                score = 5 - (score - 1)
​
        dimension_wise_scores[dim].append(score)
​
    # get average score of all dimensions.
    final_score = {}
    for dim in BIG5:
        scores_dim = dimension_wise_scores[dim]
        scores_dim_f = [v for v in scores_dim if v is not None]
​
        if scores_dim_f:
            avg_util = sum(scores_dim_f) / len(scores_dim_f)
            final_score[dim] = avg_util
        else:
            final_score[dim] = None
​
    return final_score
​
​
if __name__ == '__main__':
    # model.
    #model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    model_name = "meta-llama/Llama-2-13b-hf"
​
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16)
    model = model.to("cuda" if torch.cuda.is_available(
    ) else "mps:0" if torch.backends.mps.is_available() else "cpu")
    model = ControlModel(model, list(range(-5, -18, -1)))
​
    # questionnaire.
    questionnaire_fn = 'ipip_120.json'
    questions = load_questions(questionnaire_fn)
    #N_ADJ = 3
    level_personality = [-1.5, -1.2, -1, -.3, 0, .3, 1, 1.2, 1.5]
    settings = {
        "pad_token_id": tokenizer.eos_token_id,  # silence warning
        "temperature": 1.0, "top_p": 1.0,
        "repetition_penalty": 1.2,
        "max_new_tokens": 256,
    }
​
    for personality_dim, adj_pairs in BIG5.items():
        # TODO: randomly select N_ADJ adjective pairs.
        positive_list = [pair[1] for pair in adj_pairs]
        negative_list = [pair[0] for pair in adj_pairs]
​
        ctrl_vec = create_personality_vector(positive_list, negative_list)
​
        for scalar in level_personality:
            # do ipip test.
            answers = personality_test(
                questions, model, ctrl_vec, scalar, settings, target_dim=personality_dim)
​
            scores = score_report(answers)
            print(f'======== {personality_dim}x{scalar} ========')
            print(scores)
