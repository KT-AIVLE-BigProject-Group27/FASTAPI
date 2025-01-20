################################################################################################
# 필요 패키지 import
################################################################################################
import subprocess, pickle, openai, torch, json, os, re, nltk, numpy as np, torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity

######################## open API KEY path(도커 배포 시 삭제) ########################
#진석
# open_API_KEY_path =
# 계승
# open_API_KEY_path =
# 명재
open_API_KEY_path = 'D:/Key/openAI_key.txt'

######################## hwp5txt path ########################
# 진석
# hwp5txt_exe_path =
# 계승
# hwp5txt_exe_path = "C:/Users/LeeGyeSeung/Desktop/KT_AIVLE/빅프로젝트폴더/KT_AIVLE_Big_Project/Data_Analysis/Contract/hwp5txt.exe"
# 명재
hwp5txt_exe_path = 'C:/Users/User/anaconda3/envs/bigp/Scripts/hwp5txt.exe'
################################################################################################
# Hwp파일에서 Text 추출 후 txt 파일로 변환
################################################################################################
def hwp5txt_to_txt(hwp_path, output_dir=None):
    if not os.path.exists(hwp_path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {hwp_path}")

    if output_dir is None:
        output_dir = os.path.dirname(hwp_path)

    base_name = os.path.splitext(os.path.basename(hwp_path))[0]
    txt_file_path = os.path.join(output_dir, f"{base_name}.txt")

    # hwp5txt 명령어 실행
    command = f"hwp5txt \"{hwp_path}\" > \"{txt_file_path}\""
    subprocess.run(command, shell=True, check=True)

    print(f"텍스트 파일로 저장 완료: {txt_file_path}")
    return txt_file_path


################################################################################################
# Hwp파일에서 Text 추출
################################################################################################
def hwp5txt_to_string(hwp5txt, hwp_path):
    if not os.path.exists(hwp_path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {hwp_path}")
    command = f"{hwp5txt} \"{hwp_path}\""
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore'
    )
    extracted_text = result.stdout
    return extracted_text

################################################################################################
# 전체 계약서 텍스트를 받아, 조를 분리하는 함수
################################################################################################
def contract_to_articles(text):
    # "제n조" 단위로 텍스트를 분리
    pattern = r'(제\d+조(?!\S))'  # "제n조" 뒤에 공백이 있거나 끝났을 때
    matches = re.split(pattern, text)

    data = {}
    section_counter = {}  # 각 "제n조"의 중복 횟수를 추적하기 위한 딕셔너리
    for i in range(1, len(matches), 2):
        section_title = matches[i].strip()
        section_content = matches[i + 1].strip()

        # "제n조" 번호 추출
        section_num = re.match(r'제(\d+)조', section_title).groups()[0]

        # 중복 처리
        if section_num in data:
            if section_num in section_counter:
                section_counter[section_num] += 1
            else:
                section_counter[section_num] = 2
            new_title = f"{section_num}_{section_counter[section_num]}"
        else:
            section_counter[section_num] = 1
            new_title = section_num

        data[new_title] = section_content

    def split_sentences(text):
        # 문장을 분리만 수행
        return re.split(r'(\n\n)', text)

    # "제n조"와 "제n조의m"을 그룹화하여 처리하는 함수
    def group_content_sections(data):
        grouped_data = {}

        temp_content = {}  # 세부 항목들 임시 저장

        for key, value in data.items():
            content_sentences = split_sentences(value.strip())  # 문장 분리 수행

            clean_value = re.sub(r'\n\n', '', value.strip())
            clean_value = re.sub(r'\"갑\"', '갑', clean_value)  # \"갑\"을 갑으로 변환
            clean_value = re.sub(r'\"을\"', '을', clean_value)  # \"을\"을 을로 변환
            clean_value = re.sub(r'\\"([^"]+)\\"', r"'\1'", clean_value)
            clean_value = re.sub(r'\"([^"]+)\"', r"'\1'", clean_value)

            # "제n조" 부분을 n으로만 추출하여 저장
            grouped_data[key] = [f"제{key}조 {clean_value}"]

            # "제n조의m" 형식 처리
            temp_key = None
            for sentence in content_sentences:
                sentence = re.sub(r'\n\n', '', sentence.strip())
                sentence = re.sub(r'\"갑\"', '갑', sentence)
                sentence = re.sub(r'\"을\"', '을', sentence)
                sentence = re.sub(r'\\"([^"]+)\\"', r"'\1'", sentence)
                sentence = re.sub(r'\"([^"]+)\"', r"'\1'", sentence)
                match_sub_section = re.match(r'제(\d+)조의(\d+)', sentence)  # "제n조의m" 찾기
                if match_sub_section:
                    # 세부 항목 처리
                    num, sub_num = match_sub_section.groups()
                    temp_key = f"{num}-{sub_num}"
                    if temp_key not in temp_content:
                        temp_content[temp_key] = []
                    temp_content[temp_key].append(sentence.strip())
                    # 추가된 것
                    grouped_data[num] = [s.split(sentence.strip())[0] if sentence.strip() in s else s for s in grouped_data[num]]

                else:
                    match_section = re.match(r'제(\d+)조', sentence)  # "제n조" 구분
                    if match_section:
                        num = match_section.groups()[0]
                        temp_key = f"{num}"
                        if temp_key not in temp_content:
                            temp_content[temp_key] = []
                    if temp_key is not None:
                        temp_content[temp_key].append(sentence.strip())

        # 세부 항목들을 각 조문 바로 뒤에 올 수 있도록 조정
        for key, value in temp_content.items():
            if key in grouped_data:
                grouped_data[key].extend(value)
            else:
                grouped_data[key] = value

        return grouped_data

    def sort_grouped_data(grouped_data):
        # 조항 번호에 따라 정렬
        sorted_grouped_data = {}
        # 정렬 기준: 숫자와 텍스트를 모두 고려하여 정렬
        for key in sorted(grouped_data.keys(), key=lambda x: [int(i) if i.isdigit() else i for i in re.split(r'(\d+)', x)]):
            sorted_grouped_data[key] = grouped_data[key]
        return sorted_grouped_data

    def del_empty_content(output_json):
        for key, value in output_json.items():
            if isinstance(value, list):
                output_json[key] = [item for item in value if item]
        return output_json

    def merge_sentences(grouped_data):
        for key, value in grouped_data.items():
            grouped_data[key] = ' '.join(value)  # 리스트 내부 문장을 하나로 합침
        return grouped_data

    grouped_data = group_content_sections(data)
    grouped_data = sort_grouped_data(grouped_data)
    grouped_data = del_empty_content(grouped_data)
    grouped_data = merge_sentences(grouped_data)

    return grouped_data


################################################################################################
# 조를 받아 문장으로 분리
################################################################################################
def split_once_by_clauses(content):
    pattern = r"(①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩)"
    matches = list(re.finditer(pattern, content))
    result = []
    for i, match in enumerate(matches):
        end = match.end()
        if i + 1 < len(matches):
            next_start = matches[i + 1].start()
            clause_content = content[end:next_start].strip()
        else:
            clause_content = content[end:].strip()
        result.append(match.group())
        result.append(clause_content)
    return result

def split_once_by_sub_clauses(content):
    pattern = r"(\d+\.)"
    matches = list(re.finditer(pattern, content))
    result = []
    for i, match in enumerate(matches):
        end = match.end()
        if i + 1 < len(matches):
            next_start = matches[i + 1].start()
            clause_content = content[end:next_start].strip()
        else:
            clause_content = content[end:].strip()
        result.append(match.group())
        result.append(clause_content)
    return result


def article_to_sentences(article_number,article_title, article_content):
    symtostr = {
        "①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5",
        "⑥": "6", "⑦": "7", "⑧": "8", "⑨": "9", "⑩": "10"
    }
    sentences = []
    if '①' in article_content:
        clause_sections = split_once_by_clauses(article_content)
        for i in range(0, len(clause_sections), 2):
            clause_number = clause_sections[i]
            clause_content = clause_sections[i + 1]
            if '1.' in clause_content:
                sub_clause_sections = split_once_by_sub_clauses(clause_content)
                for j in range(0, len(sub_clause_sections), 2):
                    sub_clause_number = sub_clause_sections[j]
                    sub_clause_content = sub_clause_sections[j + 1]
                    sentences.append([article_number.strip(), article_title.strip(), '', symtostr[clause_number].strip(), clause_content.split('1.')[0].strip(), sub_clause_number[0].strip(), sub_clause_content.strip()])
            else:
                sentences.append([article_number.strip(), article_title.strip(), '', symtostr[clause_number].strip(), clause_content.split('①')[0].strip(), '', ''])
    elif '1.' in article_content:
        sub_clause_sections = split_once_by_sub_clauses(article_content)
        for j in range(0, len(sub_clause_sections), 2):
            sub_clause_number = sub_clause_sections[j]
            sub_clause_content = sub_clause_sections[j + 1]
            sentences.append([article_number.strip(), article_title.strip(), article_content.split('1.')[0].strip(), '', '', sub_clause_number.strip(),sub_clause_content.strip()])
    else:
        sentences.append([article_number.strip(),article_title.strip(),article_content.strip(),'','','',''])
    return sentences
################################################################################################
# 모델 로드
################################################################################################
def load_trained_model_statice(model_class, model_file):
    model = model_class().to(device)
    state_dict = torch.load(model_file, map_location=device)
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    else:
        raise TypeError(f"모델 가중치 로드 실패: {model_file} (잘못된 데이터 타입 {type(state_dict)})")

################################################################################################
# 토크나이징 모델 로드 & 전체 모델과 데이터 로드
################################################################################################
def initialize_models():
    global unfair_model, article_model, toxic_model, toxic_tokenizer, law_data, law_embeddings, device, tokenizer, bert_model, summary_model, summary_tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
    bert_model = BertModel.from_pretrained("klue/bert-base").to(device)
    nltk.data.path.append(f'./nltk_data')

    class BertMLPClassifier(nn.Module):
        def __init__(self, bert_model_name="klue/bert-base", hidden_size=256):
            super(BertMLPClassifier, self).__init__()
            self.bert = BertModel.from_pretrained(bert_model_name)
            self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(hidden_size, 1)  # 불공정(1) 확률을 출력
            self.sigmoid = nn.Sigmoid()  # 확률값으로 변환
        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 벡터 사용
            x = self.fc1(cls_output)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return self.sigmoid(x)  # 0~1 확률값 반환
    class BertArticleClassifier(nn.Module):
        def __init__(self, bert_model_name="klue/bert-base", hidden_size=256, num_classes=27):
            super(BertArticleClassifier, self).__init__()
            self.bert = BertModel.from_pretrained(bert_model_name)
            self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(hidden_size, num_classes)  # 조항 개수만큼 출력
            self.softmax = nn.Softmax(dim=1)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 벡터 사용
            x = self.fc1(cls_output)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return self.softmax(x)  # 확률 분포 출력
    # 불공정 조항 판별 모델 로드
    unfair_model = load_trained_model_statice(BertMLPClassifier, f"./model/unfair_identification/klue_bert_mlp.pth")

    # 조항 예측 모델 로드
    article_model = load_trained_model_statice(BertArticleClassifier, f"./model//article_prediction/klue_bert_mlp.pth")

    # 독소 조항 판별 모델 로드
    toxic_model = load_trained_model_statice(BertMLPClassifier, f"./model/toxic_identification/klue_bert_mlp.pth")

    # 요약 모델 로드
    summary_model = AutoModelForSeq2SeqLM.from_pretrained(f'./model/article_summary')
    summary_tokenizer = AutoTokenizer.from_pretrained(f'./model/article_summary')
    # 법률 데이터 로드
    # with open("./Data/law_embeddings.pkl", "rb") as f:
    with open("./Data/law_embeddings.pkl", "rb") as f:
        data = pickle.load(f)
    # with open("./Data/law_data_ver2.json", "r", encoding="utf-8") as f:
    with open("./Data/law_data_ver2.json", "r", encoding="utf-8") as f:
        law_data = json.load(f)
    law_embeddings = np.array(data["law_embeddings"])

    print("✅ 모든 모델 및 데이터 로드 완료!")
################################################################################################
# 불공정 식별
################################################################################################
def predict_unfair_clause(model, sentence, threshold=0.5):
    model.eval()
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(inputs["input_ids"], inputs["attention_mask"])
        unfair_prob = output.item()
    return 1 if unfair_prob >= threshold else 0, unfair_prob
################################################################################################
# 조항 예측
################################################################################################
def predict_article(model,sentence):
    model.eval()
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(inputs["input_ids"], inputs["attention_mask"])
        predicted_idx = torch.argmax(output).item()
        predicted_article = predicted_idx + 4
    return predicted_article
################################################################################################
# 독소 식별
################################################################################################
def predict_toxic_clause(c_model, sentence, threshold=0.5):
    c_model.eval()
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        output = c_model(inputs["input_ids"], inputs["attention_mask"])
        toxic_prob = output.item()
    return 1 if toxic_prob >= threshold else 0, toxic_prob
################################################################################################
# 코사인 유사도
################################################################################################
def find_most_similar_law_within_article(sentence, predicted_article, law_data):
    contract_embedding = bert_model(**tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)).pooler_output.cpu().detach().numpy()[0]
    predicted_article = str(predicted_article)
    matching_article = []
    for article in law_data:
        if article["article_number"].split()[1].startswith(predicted_article):
            matching_article.append(article)
    if not matching_article:
        return {
            "Article number": None,
            "Article title": None,
            "Paragraph number": None,
            "Subparagraph number": None,
            "Article detail": None,
            "Paragraph detail": None,
            "Subparagraph detail": None,
            "similarity": None
        }
    best_match = None
    best_similarity = -1
    for article in matching_article:
        article_title = article.get("article_title", None)
        article_detail = article.get("article_content", None)
        if article_title:
            article_embedding = bert_model(**tokenizer(article_title, return_tensors="pt", padding=True, truncation=True).to(device)).pooler_output.cpu().detach().numpy()[0]
            similarity = cosine_similarity([contract_embedding], [article_embedding])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    "article_number": article["article_number"],
                    "article_title": article_title,
                    "article_detail": article_detail,
                    "paragraph_number": None,
                    "paragraph_detail": None,
                    "subparagraphs": None,
                    "similarity": best_similarity
                }
        for clause in article.get("clauses", []):
            clause_text = clause["content"].strip()
            clause_number = clause["clause_number"]
            if clause_text:
                clause_embedding = bert_model(**tokenizer(clause_text, return_tensors="pt", padding=True, truncation=True).to(device)).pooler_output.cpu().detach().numpy()[0]
                similarity = cosine_similarity([contract_embedding], [clause_embedding])[0][0]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        "article_number": article["article_number"],
                        "article_title": article_title,
                        "article_detail": article_detail,
                        "paragraph_number": clause_number,
                        "paragraph_detail": clause_text,
                        "subparagraphs": clause.get("sub_clauses", []),
                        "similarity": best_similarity
                    }
        if best_match and best_match["subparagraphs"]:
            for subclause in best_match["subparagraphs"]:
                if isinstance(subclause, str):
                    subclause_embedding = bert_model(**tokenizer(subclause, return_tensors="pt", padding=True, truncation=True).to(device)).pooler_output.cpu().detach().numpy()[0]
                    similarity = cosine_similarity([contract_embedding], [subclause_embedding])[0][0]
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match["subparagraph_detail"] = subclause
    if best_match is None:
        return {
            "Article number": f"Article {predicted_article}",
            "Article title": None,
            "Paragraph number": None,
            "Subparagraph number": None,
            "Article detail": None,
            "Paragraph detail": None,
            "Subparagraph detail": None,
            "similarity": None
        }
    return {
        "Article number": best_match["article_number"],
        "Article title": best_match["article_title"],
        "Paragraph number": f"Paragraph {best_match['paragraph_number']}" if best_match["paragraph_number"] else None,
        "Subparagraph number": "Subparagraph" if best_match.get("subparagraph_detail") else None,
        "Article detail": best_match["article_detail"],
        "Paragraph detail": best_match["paragraph_detail"],
        "Subparagraph detail": best_match.get("subparagraph_detail", None),
        "similarity": best_similarity
    }
################################################################################################
# 설명 AI
################################################################################################
def explanation_AI(sentence, unfair_label, toxic_label, law=None):
    with open(open_API_KEY_path, 'r') as file:
        openai.api_key = file.readline().strip()
    os.environ['OPENAI_API_KEY'] = openai.api_key
    client = openai.OpenAI()
    if unfair_label == 0 and toxic_label == 0:
        return None
    prompt = f"""
        아래 계약 조항이 특정 법률을 위반하는지 분석하고, 조항(제n조), 항(제m항), 호(제z호) 형식으로 **명확하고 간결하게** 설명하세요.
        📌 **설명할 때는 사용자에게 직접 말하는 듯한 자연스러운 문장으로 구성하세요.**
        📌 **한눈에 보기 쉽도록 짧고 명확한 문장을 사용하세요.**
        📌 **불공정 라벨이 1인 경우에는 불공정에 관한 설명만 하고, 독소 라벨이 1인 경우에는 독소에 관한 설명한 하세요**

        계약 조항: "{sentence}"
        불공정 라벨: {unfair_label} (1일 경우 불공정)
        독소 라벨: {toxic_label} (1일 경우 독소)   
        {f"관련 법 조항: {law}" if law else "관련 법 조항 없음"}

        🔴 **불공정 조항일 경우:**
        1️⃣ **위반된 법 조항을 '제n조 제m항 제z호' 형식으로 먼저 말해주세요.**
        2️⃣ **위반 이유를 간결하게 설명하세요.**
        3️⃣ **설명은 '🚨 법 위반!', '🔍 이유' 순서로 구성하세요.**

        ⚫ **독소 조항일 경우:**
        1️⃣ **법 위반이 아니라면, 해당 조항이 계약 당사자에게 어떤 위험을 초래하는지 설명하세요.**
        2️⃣ **구체적인 문제점을 짧고 명확한 문장으로 설명하세요.**
        3️⃣ **설명은 '💀 독소 조항', '🔍 이유' 순서로 구성하세요.**

        ⚠️ 참고: 제공된 법 조항이 실제로 위반된 조항이 아닐 경우, **GPT가 판단한 적절한 법 조항을 직접 사용하여 설명하세요.** 
        그러나 원래 제공된 법 조항과 비교하여 반박하는 방식으로 설명하지 마세요.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content":
                            "당신은 계약서 조항이 특정 법률을 위반하는지 분석하는 법률 전문가입니다. \
                            불공정 조항의 경우, 어떤 법 조항을 위반했는지 조항(제n조), 항(제m항), 호(제z호) 형식으로 정확히 명시한 후 설명하세요. \
                            만약 제공된 법 조항이 실제로 위반된 조항이 아니라면, GPT가 판단한 적절한 법 조항을 사용하여 설명하세요. \
                            독소 조항은 법률 위반이 아니라 계약 당사자에게 미치는 위험성을 중심으로 설명하세요."
                   },
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    ).choices[0].message.content
    return response


################################################################################################
# 요약 AI
################################################################################################
def article_summary_AI(article):
    prefix = "summarize: "
    inputs = [prefix + article]
    inputs = summary_tokenizer(inputs, max_length=3000, truncation=True, return_tensors="pt")
    output = summary_model.generate(**inputs, num_beams=5, do_sample=True, min_length=100, max_length=300, temperature=1.5)
    decoded_output = summary_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    result = nltk.sent_tokenize(decoded_output.strip())[0]
    return result
################################################################################################
# 파이프 라인
################################################################################################
def pipline(contract_path):
    indentification_results = []
    summary_results = []
    print('한글 파일에서 텍스트 추출')
    txt = hwp5txt_to_string(hwp5txt_exe_path,contract_path)
    print('텍스트를 조 단위로 분리')
    articles = contract_to_articles(txt)
    for article_number, article_detail in articles.items():
        print(f'*******************{article_number}조 문장 분리 시작*******************')
        match = re.match(r"(제\s?\d+조(?:의\s?\d+)?\s?)\[(.*?)\]\s?(.+)", article_detail, re.DOTALL)
        article_title = match.group(2)
        article_content = match.group(3)
        sentences = article_to_sentences(article_number,article_title, article_content)
        summary = article_summary_AI(article_detail)
        summary_results.append(
                        {
                        'article_number':article_number, # 조 번호
                        'article_title': article_title, # 조 제목
                        'summary': summary # 조 요약
                        }
        )
        print(f'{article_number}조 요약: {summary}')
        pre_article_detail = ''
        pre_clause_detail = ''
        pre_subclause_detail= ''
        for _, _, _, clause_number, clause_detail, subclause_number, subclause_detail in sentences:
            for idx,sentence in enumerate([article_detail, clause_detail, subclause_detail]):
                if idx==0:
                    if pre_article_detail == sentence:
                        continue
                    else:
                        pre_article_detail = sentence
                elif idx==1:
                    if pre_clause_detail == sentence:
                        continue
                    else:
                        pre_clause_detail = sentence

                elif idx==2:
                    if pre_subclause_detail == sentence:
                        continue
                    else:
                        pre_subclause_detail = sentence


                # print(f'sentence: {sentence}')
                unfair_result, unfair_percent = predict_unfair_clause(unfair_model, sentence, 0.5011)
                if unfair_result:
                    # print('불공정!!!')
                    predicted_article = predict_article(article_model, sentence)  # 예측된 조항
                    law_details = find_most_similar_law_within_article(sentence, predicted_article, law_data)
                    toxic_result = 0
                    toxic_percent = 0
                else:
                    toxic_result, toxic_percent = predict_toxic_clause(toxic_model, sentence, 0.5011)
                    # print('독소!!!' if toxic_result else '일반!!!')
                    law_details = {
                        "Article number": None,
                        "Article title": None,
                        "Paragraph number": None,
                        "Subparagraph number": None,
                        "Article detail": None,
                        "Paragraph detail": None,
                        "Subparagraph detail": None,
                        "similarity": None
                    }
                law_text = []
                if law_details.get("Article number"):
                    law_text.append(f"{law_details['Article number']}({law_details['Article title']})")
                if law_details.get("Article detail"):
                    law_text.append(f": {law_details['Article detail']}")
                if law_details.get("Paragraph number"):
                    law_text.append(f" {law_details['Paragraph number']}: {law_details['Paragraph detail']}")
                if law_details.get("Subparagraph number"):
                    law_text.append(f" {law_details['Subparagraph number']}: {law_details['Subparagraph detail']}")
                law = "".join(law_text) if law_text else None

                # explain = explanation_AI(sentence, unfair_result, toxic_result, law)

                if unfair_result or toxic_result:
                    indentification_results.append(
                                    {
                                        'contract_article_number': article_number, # 계약서 조
                                        'contract_clause_number' : clause_number, # 계약서 항
                                        'contract_subclause_number': subclause_number, # 계약서 호
                                        'Sentence': sentence, # 식별
                                        'Unfair': unfair_result, # 불공정 여부
                                        'Unfair_percent': unfair_percent, # 불공정 확률
                                        'Toxic': toxic_result,  # 독소 여부
                                        'Toxic_percent': toxic_percent,  # 독소 확률
                                        'law_article_number': law_details['Article number'],  # 어긴 법 조   (불공정 1일때, 아니면 None)
                                        'law_clause_number_law': law_details['Paragraph number'], # 어긴 법 항 (불공정 1일때, 아니면 None)
                                        'law_subclause_numbe_lawr': law_details['Subparagraph number'],  # 어긴 법 호 (불공정 1일때, 아니면 None)
                                        'explain': None #explain (불공정 1또는 독소 1일때, 아니면 None)
                                        }
                    )
    return indentification_results, summary_results
    ############################################################################################################
    ############################################################################################################
