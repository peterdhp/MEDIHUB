import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.output_parsers import StrOutputParser
import warnings
import re
import pandas as pd
import numpy as np

from langchain_community.callbacks import get_openai_callback


warnings.filterwarnings("ignore", message="Data Validation extension is not supported and will be removed")



openai_api_key = st.secrets["OPENAI_API_KEY"]
EXCEL_FILE = 'LLM_DB_ver6.22_inclusion_0422.xlsx'
MED_SP = ['Gastroenterology_pain(+)', 'Gastroenterology_pain(-)', 'Infectology_fever(+)', 'Infectology_fever(-)', 'Endocrinology_DiabeticMellitus',
 'Endocrinology_Thyroid&Pituitary', 'Rheumatology', 'Nephrology_edema', 'Nephrology_general', 'Hematology', 'Neurology_headache',
 'Neurology_CVA',  'Neurology_dyskinesia',  'Orthopedics_Spine_Lower back',  'Orthopedics_Spine_Neck',  'Orthopedics_knee',  'Orthopedics_hip', 'Orthopedics_shoulder',
 'Orthopedics_elbow',  'Orthopedics_wrist&fingers',  'Orthopedics_foot', 'Cardiology_pain',  'Cardiology_palpitations',  'Pulmonology_cough&sputum',
 'Pulmonology_dyspnea', 'Urology_Sexual problem',  'Urology_Voiding Dysfunction',  'Allergy',  'General Surgery_Anus', 'General Surgery_Thyroid',
 'General Surgery_Hepatopancreato', 'General Surgery_Breast',  'Obstetrics',  'Gynecology_Menstrual',  'Gynecology_Infection',  'Dermatology',
 'Otorhinolaryngology_Nose',  'Otorhinolaryngology_Ear', 'Otorhinolaryngology_Throat', 'Ophthalmology','Psychiatry_Mood', 'Psychiatry_Anxiety',
 'Psychiatry_Personality', 'Psychiatry_Memory', 'Psychiatry_Child', 'Dizziness']



def question_retriever(CC_type):
    excel_dict = load_excel(file=EXCEL_FILE, sheet_white_list=MED_SP)
    hx_sheet = excel_dict[CC_type]

    ############################################### preapre history taking
    ### subset_ sheet
    hx_sheet = clean_sheet(hx_sheet) # 데이터로 사용할 가로세로 영역 지정
    qa = hx_sheet.loc[:,['질문','질문_영문', '선택유형', '선택지','선택지_영문(+)']] # 5가지 column만 사용
    qa.columns = ['q_kor', 'q_eng', 'q_type', 'a_kor', 'a_eng']
    qa.loc[:,'q_kor'] = qa.loc[:,'q_kor'].ffill() # 질문 빈칸 아래로 복사(셀합치기는 첫칸에만 텍스트 있는 것으로 인식)
    qa.loc[:,'q_eng'] = qa.loc[:,'q_eng'].ffill()
    qa.loc[:,'q_type'] = qa.loc[:,'q_type'].ffill()

    ### group by question
    ### 질문 column이 unique 하도록 답변을 list로 합침
    qa.loc[:, 'a_kor'] = qa['a_kor'].fillna('')
    qa.loc[:, 'a_eng'] = qa['a_eng'].fillna('')
    qa_groupby = qa.groupby(['q_kor', 'q_eng', 'q_type'], as_index=False, sort=False)
    qa_groupby_a_kor = qa_groupby['a_kor'].apply(list)
    qa_groupby_a_eng = qa_groupby['a_eng'].apply(list)
    qa_choice = pd.merge(left=qa_groupby_a_kor, right=qa_groupby_a_eng, on=['q_kor', 'q_eng', 'q_type'])
    qa_choice = qa_choice.rename(columns={'a_kor':'a_kor_list', 'a_eng':'a_eng_list'})
    
    
    return '\n'.join(qa_choice.apply(format_row_korQA, axis=1))

def format_row_korQA(row):
    answers = ', '.join([str(i) for i in row['a_kor_list']])
    return f"{row['q_kor']}\texample answers : {answers}"        
        
def LLM_respond(msg_log, demographics_text ,CC, CC_type, question_list):
    """채팅 내용을 기반으로 질문을 함"""
    
    
    
    system_prompt = [("system", """You are a Korean doctor. you are going to ask the patient questions about his/her symptoms, ONE AT A TIME. Only when you think you have asked enough question to diagnose the patient, please summarize what the patient has said and receive confirmation. After recieving confirmation, tell the patient to wait a few minutes before the doctor's meeting.
ALL the questions in the list MUST be asked. HOWEVER, when the patient information doesn't match the content of the question, skip the question. For example, don't ask appenditis related question for patient with appendectomy history
The list of example answers should ALWAYS be provide with the question as a bullet point list. It should look like this :

다음과 같은 증상이 있었나요? 
    - 복통
    - 설사
    - 발열
    - 구역
    - 구토
    
    [question list]
    {question_list}
    
    [patient information]
    {demographics_text}                     
    """)]
    
    #print(system_prompt)
    #print(msg_log)
    prompt_temp = system_prompt + msg_log
    #print(prompt_temp)
    
    prompt = ChatPromptTemplate.from_messages(prompt_temp)
    #print(question_list)
    
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature = 0) 
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser    
    
    output = chain.stream({"demographics_text" : demographics_text, "question_list": "\n ".join(question_list)})
    return output

def CC_classifier(CC,demographics_text):

    
    prompt_template = """Classify the patient's chief complaint based on the following details: 
Chief complaint : {CC}
Demographics : {demographics}

Please classify the chief complaint into one of the following categories:
{CC_category}

Also give me the confidence level of your output from 1 to 10

the ouput should be like the example bellow without any explanation:
(Urology_Sexual problem, 8)

"""
    prompt_class = PromptTemplate.from_template(prompt_template)
    #prompt_class.format({"CC" : CC, "CC_category" : "\n".join(MED_SP), "demographics" : demographics_text})
    
    
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature = 0) 
    output_parser = StrOutputParser()

    chain = prompt_class | llm | output_parser    
    
    
    
    output = chain.invoke({"CC" : CC, "CC_category" : "\n".join(MED_SP), "demographics" : demographics_text})
    output_tuple_temp = re.sub(r'^\s+', '',output)
    output_tuple_temp = output_tuple_temp.strip('()').split(',')
    
    output_tuple= (output_tuple_temp[0],int(output_tuple_temp[1]))
    return output_tuple

def CC_updater(CC,demographics,msg_log):
    
    demographics_text = '\n'.join([
                "주증상 : {}".format(CC),
                "생년월일 : {}".format(demographics['birth-date']),
                "성별 : {}".format(demographics['gender']),
                "흡연 : {}".format(demographics['smoking']),
                "음주 : {}".format(demographics['alcohol']),
                "기저질환  : {}".format("해당없음" if not demographics['health_conditions'] else demographics['health_conditions']),
                "추가정보 : {}".format("해당없음" if not demographics['additional_info'] else demographics['additional_info'])
                ])
    
    prompt_temp = """Classify the patient's problem based on the following details: 
Chief complaint : {CC}
Demographics : {demographics}
Chat log : 
{msg_log_text}

Please classify the patient's problem into one of the following categories:
{CC_category}

Also give me the confidence level of your output from 1 to 10

the ouput should be like the example bellow without any explanation:
(Urology_Sexual problem, 8)
"""
    prompt_class = PromptTemplate.from_template(prompt_temp)
    
    user_type_mapping = {'human': '[patient]', 'ai': '[doctor]'}
    msg_log_text = "\n".join(f"{user_type_mapping[sender]} : {message}" for sender, message in msg_log)
    
    
    
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature = 0) 
    output_parser = StrOutputParser()

    chain = prompt_class | llm | output_parser    
    
    output = chain.invoke({"CC" : CC, "CC_category" : "\n".join(MED_SP), "demographics" : demographics_text,"msg_log_text":msg_log_text})
    output_tuple_temp = re.sub(r'^\s+', '',output)
    output_tuple_temp = output_tuple_temp.strip('()').split(',')
    
    output_tuple= (output_tuple_temp[0],int(output_tuple_temp[1]))
    return output_tuple

def CC_updater_GPT4(CC,demographics_text,msg_log):
    
    
    prompt_temp = """Classify the patient's problem based on the following details: 
Chief complaint : {CC}
Demographics : {demographics}
Chat log : 
{msg_log_text}

Please classify the patient's problem into one of the following categories:
{CC_category}

Also give me the confidence level of your output from 1 to 10

the ouput should be like the example bellow without any explanation:
(Urology_Sexual problem, 8)
"""
    prompt_class = PromptTemplate.from_template(prompt_temp)
    
    user_type_mapping = {'human': '[patient]', 'ai': '[doctor]'}
    msg_log_text = "\n".join(f"{user_type_mapping[sender]} : {message}" for sender, message in msg_log)
    
    
    
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature = 0) 
    output_parser = StrOutputParser()

    chain = prompt_class | llm | output_parser    
    
    output = chain.invoke({"CC" : CC, "CC_category" : "\n".join(MED_SP), "demographics" : demographics_text,"msg_log_text":msg_log_text})
    output_tuple_temp = re.sub(r'^\s+', '',output)
    output_tuple_temp = output_tuple_temp.strip('()').split(',')
    
    output_tuple= (output_tuple_temp[0],int(output_tuple_temp[1]))
    return output_tuple
    
def survey_summary(msg_log, demographics ,CC, CC_type):
    """문진 내용을 기반으로 질문을 함"""
    
    demographics_text = '\n'.join([
                "주증상 : {}".format(CC),
                "생년월일 : {}".format(demographics['birth-date']),
                "성별 : {}".format(demographics['gender']),
                "흡연 : {}".format(demographics['smoking']),
                "음주 : {}".format(demographics['alcohol']),
                "기저질환  : {}".format("해당없음" if not demographics['health_conditions'] else demographics['health_conditions']),
                "추가정보 : {}".format("해당없음" if not demographics['additional_info'] else demographics['additional_info'])
                ])
    
    
    
    prompt_template = """Given the following information and chat log, diagnose the patient. 
[patient information]
{demographics_text}

[chat log]
{msg_log_text}

Give 5 possible diagnosis and appropiate tests. Also if there are important issues that should be double checked. please point it out.
The output should be in Korean and follow the format below :

[예상 진단]
1. diagnosis 1 
: reason for diagnosis 1
: tests that should be carried out to confirm diagnosis 1

2. diagnosis 2
: reason for diagnosis 2
: tests that should be carried out to confirm diagnosis 2

...
[주의 할 점]
list of very important issues to point out

-----

"""
    
    user_type_mapping = {'human': '[patient]', 'ai': '[doctor]'}
    msg_log_text = "\n".join(f"{user_type_mapping[sender]} : {message}" for sender, message in msg_log)
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature = 0) 
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser    
    
    output = chain.invoke({"demographics_text" : demographics_text,"msg_log_text" : msg_log_text})
    return output




if 'CC' not in st.session_state:
    st.switch_page('app_streamlit.py')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    role = '🩺' if message[0] == 'ai' else message[0]
    with st.chat_message(role):
        st.markdown(message[1])
        
if "total_cost" not in st.session_state:
    st.session_state.totalcost = 0

#print(question_list)

if len(st.session_state.messages) ==0 : 
    


    with get_openai_callback() as cb:
        (st.session_state.CC_type,st.session_state.CC_confidence) = CC_classifier(st.session_state.CC,st.session_state.demographics_text)
        print('\nCC_classification')
        st.session_state.totalcost += cb.total_cost
        print(st.session_state.CC_type)
        print(st.session_state.CC_confidence)
        print(cb)
        
    st.session_state.question_list = question_retriever(st.session_state.CC_type)
    
    st.toast(('CC classification : "%s, Confidence : %s/10"'%(st.session_state.CC_type,str(st.session_state.CC_confidence))))

    if st.session_state.CC_confidence >= 9: 
        with st.chat_message("🩺"):
            with get_openai_callback() as cb:
                print('\nCHAT')
                stream = LLM_respond(st.session_state.messages, st.session_state.demographics_text ,st.session_state.CC, st.session_state.CC_type, st.session_state.question_list )
                st.session_state.totalcost += cb.total_cost
                print(cb)
            response = st.write_stream(stream)
        st.session_state.messages.append(("ai", response))
    else :
        st.session_state.messages.append(('ai','증상에 대해서 조금만 더 자세히 설명해주시겠어요? 동반된 증상들을 가능한 많이 알려주시면 진료에 도움이 됩니다.'))
        with st.chat_message("🩺"):
            st.markdown('증상에 대해서 조금만 더 자세히 설명해주시겠어요? 동반된 증상들을 가능한 많이 알려주시면 진료에 도움이 됩니다.')


if not st.session_state.CC_confidence >= 9:
    if len(st.session_state.messages) % 2 == 0:
        CC_type_prev = st.session_state.CC_type
        with get_openai_callback() as cb:
            print('\nCC_update')
            (st.session_state.CC_type,st.session_state.CC_confidence) = CC_updater_GPT4(st.session_state.CC,st.session_state.demographics_text,st.session_state.messages)
            st.session_state.totalcost += cb.total_cost
            print(cb)
        st.session_state.question_list  = question_retriever(st.session_state.CC_type)
        if not st.session_state.CC_type == CC_type_prev :
            st.toast('CC_classification이 %s 에서 %s으로 변경되었습니다. %s/10'%(CC_type_prev,st.session_state.CC_type,st.session_state.CC_confidence))  
        print(st.session_state.CC_type)
        print(st.session_state.CC_confidence)       
        
    if len(st.session_state.messages) % 10 == 0:
        st.session_state.CC_type_prev = st.session_state.CC_type
        with get_openai_callback() as cb:
            print('\nCC_update')
            (st.session_state.CC_type,st.session_state.CC_confidence) = CC_updater_GPT4(st.session_state.CC,st.session_state.demographics_text,st.session_state.messages)
            st.session_state.totalcost += cb.total_cost
            print(cb)
        st.session_state.question_list  = question_retriever(st.session_state.CC_type)
        if not st.session_state.CC_type == st.session_state.CC_type_prev :
            st.toast('CC_classification이 %s 에서 %s으로 변경되었습니다. %s/10'%(CC_type_prev,st.session_state.CC_type,st.session_state.CC_confidence))  
        print(st.session_state.CC_type)
        print(st.session_state.CC_confidence) 
        



# Accept user input
if userinput := st.chat_input("주어진 예시를 참고하여 답변해주세요"):
    # Add user message to chat history
    st.session_state.messages.append(("human", userinput))
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(userinput)
        
    # Display assistant response in chat message container
    with st.chat_message("🩺"):
        with get_openai_callback() as cb:
            print('\nCHAT')
            stream = LLM_respond(st.session_state.messages, st.session_state.demographics_text ,st.session_state.CC, st.session_state.CC_type, st.session_state.question_list)
            st.session_state.totalcost += cb.total_cost
            print(cb)
        response = st.write_stream(stream)
    st.session_state.messages.append(("ai", response))
    
    
#print(st.session_state.messages)   
    
def demo():
    st.session_state.status = "historytaking"
    st.session_state.messages = [('ai', '현재 불편한 부위를 구체적으로 기술해 주세요.\n예시 답변:\n- 양쪽 어깨\n- 오른쪽 어깨\n- 왼쪽 어깨'), ('human', '왼쪽 어깨'), ('ai', '다른 아픈 곳이 또 있습니까?\n예시 답변:\n- 없음\n- 다른 부위 통증 있음'), ('human', '아니요 없습니다'), ('ai', '통증의 양상은 어떤가요?\n예시 답변:\n- 뻣뻣함\n- 쑤심\n- 찌르는 듯\n- 예리함\n- 욱신욱신'), ('human', '뻣뻣하고 쑤시고 욱신욱식할 때도 있어요'), ('ai', '증상이 나타나게 된 요인은 무엇입니까?\n예시 답변:\n- 특이한 원인 없이 증상 발현\n- 운동\n- 교통사고\n- 넘어짐'), ('human', '자고 일어났더니 그냥 아프기 시작했어요'), ('ai', '다음 중 통증을 유발하는 상황을 모두 표시해 주세요.\n예시 답변:\n- 물체를 들어올릴 때\n- 가만히 있을 때\n- 팔을 위로 들 때\n- 어깨를 돌릴 때\n- 불편한 쪽으로 옆으로 누워 잘 때\n- 기타 (직접 입력)\n- 통증 유발 자세 없음'), ('human', '물체를 들어올릴 때, 팔을 위로 들 때랑 어깨를 돌릴 때 , 불편한 쪽으로 팔베게하고 잘 때 다 아파요'), ('ai', '다음과 같은 증상이 있나요?\n- 마비 또는 감각 이상\n- 따끔거림\n- 부종(붓기)\n- 뻣뻣함\n- 발적 또는 온기\n- 덜걱거리는 느낌'), ('human', '뻣뻣해요'), ('ai', '증상이 얼마나 자주 나타나나요?\n예시 답변:\n- 지속적\n- 간헐적'), ('human', '지속적'), ('ai', '증상이 처음 시작된 시기는 언제입니까?\n예시 답변:\n- 년 월'), ('human', '처음 시작된건 3개월 전쯤이요'), ('ai', '증상이 악화된 시기는 최근 언제입니까?\n예시 답변:\n- 1~2일 이내\n- 7일 이내\n- 2주\n- 4주\n- 6주 이내\n- 3개월 이내'), ('human', '3일 전부터 심해졌어요'), ('ai', '오늘은 통증이 어느 정도입니까?\n예시 답변:\n- 0 (약)\n- 1\n- 2\n- 3\n- 4\n- 5\n- 6\n- 7\n- 8\n- 9\n- 10 (강)'), ('human', '5 정도요'), ('ai', '가장 아플 때는 통증이 어느 정도입니까?\n예시 답변:\n- 0 (약)\n- 1\n- 2\n- 3\n- 4\n- 5\n- 6\n- 7\n- 8\n- 9\n- 10 (강)'), ('human', '8'), ('ai', '통증으로 인해 수행할 수 없는 동작이 있습니까?\n예시 답변:\n- 모든 동작 수행 가능\n- 있음'), ('human', '팔을 못 들겠어요'), ('ai', '증상 부위 사진이 있다면 보여주세요.\n예시 답변:\n- 사진 첨부하기\n- 넘어가기'), ('human', '없어요'), ('ai', '확인해 주셔서 감사합니다. 지금까지 말씀하신 증상을 요약하겠습니다:\n\n- 주 증상은 왼쪽 어깨 통증입니다.\n- 통증은 뻣뻣하고 쑤시며 욱신거립니다.\n- 자고 일어난 후부터 통증이 시작되었으며, 약 3개월 전부터 증상이 있었습니다.\n- 통증은 물체를 들어올릴 때, 팔을 위로 들 때, 어깨를 돌릴 때, 불편한 쪽으로 팔베게하고 잘 때 심해집니다.\n- 통증은 지속적이며, 최근 3일 전부터 심해졌습니다.\n- 오늘 통증 정도는 5이며, 가장 심할 때는 8입니다.\n- 통증으로 인해 팔을 드는 동작을 수행할 수 없습니다.\n\n이 내용이 맞습니까? 확인해 주시면 진료를 계속 진행하겠습니다.'), ('human', '맞습니다.'), ('ai', '확인해 주셔서 감사합니다. 잠시만 기다려 주시면 의사 선생님과의 상담을 준비하겠습니다. 조금만 기다려 주세요.')]


with st.sidebar:
    st.button("Demo",on_click=demo)
menu_with_redirect()
