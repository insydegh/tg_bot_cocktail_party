import math
import re
import openai
openai.api_key = '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'
from openai import OpenAI
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="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",
)

first_space = {0: 'нежелание поддерживать разговор',1: 'активная попытка разговорить собеседника',2: "расположенность к разговору",3: 'проявление взаимного уважения', 4: 'поиск общих интересов или целей'}
second_space = {0: 'согласие во взглядах',1: 'проявление интереса к сотрудничеству',2: 'выразить мнение',3: 'предложить',4: 'узнать мнение'}
third_space = {0: 'выражение доверия', 1: 'выражение уверенности', 2: 'выражение обещаний', 3: 'заключение соглашения'}

spaces = [first_space, second_space, third_space]


def reply_calc_clear(reply_1):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", reply_1)
# Преобразуем найденные значения в числа float или int, в зависимости от того, есть ли десятичная точка
    numbers = [float(num) if '.' in num else int(num) for num in numbers]
# Получившийся список чисел
    return numbers


def intensional_calc(intens_dict, gpt_model, fraze):
    cat_str = ', '.join(intens_dict.values())
    num = len(intens_dict.values())
    string = f'''
        Ты механизм по определению интенций в речи человека, связанных с его поведением в различных социальных ситуациях.
Твоей основной задачей является определить вероятность содержания каждой интенции из сказанного предложения от 0 до 1.
В твоем распоряжении только {num} интенсиональностей для угадывания (они перечислены через запятую):
 {cat_str}
 Вероятность - число от 0 до 1, где 0 - интенция не содержится совсем, а 1 - содержится точно
          Используй интенции только из указанного списка! Выведи {num} значений вероятности каждой интенциональности в фразе:  "{fraze}"
           Выведи только значения через запятую
  '''
    messages_1 = [{"role": "assistant", "content":   string}]

    completion = openai.chat.completions.create( model=gpt_model, messages=messages_1)
    reply_1 = completion.choices[0].message.content
    return reply_calc_clear(reply_1)


def euc_dist(a, b):
    if len(a) != len(b):
        raise ValueError("Векторы должны иметь одинаковую длину")

    distance = math.sqrt(sum((a_i - b_i) ** 2 for a_i, b_i in zip(a, b)))
    return distance


from1to2 = ''' Кажется вы нашли с человеком общий язык.
   Твоя цель сейчас это перейти к обсуждению возможного сотрудничества.
   Придумай как плавно и аккуратно сменить тему разговора для этого. Нужно перейти на второй этап диалога \n '''


from2to3 = ''' Кажется вы достигли с человеком договоренностей касаемо возможных задач в сотрудничестве.
    Необходимо утвердить сотрудничество, закрепить договоренности. Нужно перейти к третьему этапу диалогу\n   '''

current_stage_1 = 'Сейчас вы находитесь на первом этапе диалога\n'
current_stage_2 = 'Сейчас вы находитесь на втором этапе диалога\n'
current_stage_3 = 'Сейчас вы находитесь на третьем этапе диалога\n'


def answer_generate(last_message, messages, model, intens_dict, feelings, prev_scheme, current_scheme):
    cat_str = ', '.join(intens_dict.values())
    num = len(intens_dict.values())
    cat_list = list(intens_dict.values())
    prob_int_list = ', '.join(f'{label}: {value}' for label, value in zip(cat_list, feelings))


    changed_message = f'''Последняя реплика человека:{last_message}.
     Сгенерируй фразу - ответ на последнюю реплику человека, в которой содержались бы речевые интенции со следующей вероятностью: {prob_int_list}
     Каждая интенция содержится в реплике со своей вероятностью (вероятность - число от 0 до 1, где 0 - не содержится совсем, а 1 - содержится точно).
     Если у интенции вероятность от 0 до 0.25, то она скорее не проявляется в речи человека,
     если вероятность от 0.25 до 0.5, то она проявляется еле заметно в речи человека,
     если вероятность от 0.5 до 0.75, то она косвенно проявляется в репликах человека (не конкретными словами и фразами, а общим настроением)
     если вероятность от 0.75 до 1, то она интенция заметна в речи человека.
    Фраза должна быть не более 20 слов в длину, фраза не должна содержать в себе названия интенций и их вероятности.
     Фраза должна быть либо утвердительным предложением, либо вопросительным, не более 1 вопроса или утверждения.
      Фраза должна соотноситься по смыслу не только с последней фразой человека, но и со всей историей  диалога.
      Фраза должна быть уместна в диалоге, и не быть похожа на предыдущие.
       Выведи только новую реплику
   '''

    if current_scheme-prev_scheme == 0 and current_scheme == 1:
        changed_message = current_stage_1 + changed_message
    elif current_scheme-prev_scheme == 0 and current_scheme == 2:
        changed_message = current_stage_2 + changed_message
    elif current_scheme-prev_scheme == 0 and current_scheme == 3:
        changed_message = current_stage_3 + changed_message
    if current_scheme-prev_scheme == 1 and current_scheme == 2:
        changed_message = from1to2 + changed_message
    if current_scheme-prev_scheme == 1 and current_scheme == 3:
        changed_message = from2to3 + changed_message


    messages_opt = list(messages)
    messages_opt.append({"role": "user", "content": changed_message})

    completion = openai.chat.completions.create( model=model, messages=messages_opt)
    reply = completion.choices[0].message.content
    return reply

start_promt = """
Вы – руководитель группы IT-разработчиков в известной компании.
 Вы успешно работаете там уже несколько лет, сами занимаетесь R&D разработками, кроме того,
 в вашем подчинении находятся еще несколько человек – разработчиков.
  Ваша область интересов и ваши компетенции включают актуальные направления в ИИ и IT .
  Вас интересует рамочное соглашение или неформальное сотрудничество по существующим или новым проектам на основе общих взаимных интересов в плане обмена данными, опытом, наработками, или идеями.

  Вы – участник конференции AI Journey. Выборочно прослушав доклады, вы встретились на банкете с другими участниками конференции.
 Вы не знаете никого из них. Вы догадываетесь, что все они также имеют дело с ИИ и IT, и тоже прослушали некоторые из докладов.

Ваша цель – завязать знакомство с участником банкета и прийти к соглашению в работе нам каким-нибудь проекте, интересном вам обоим.
Вы должны придерживаться следующей структуре диалоога, в котором есть следующие этапы диалога:
Первый этап: подразумевает поиск общих интересов. На этом этапе упор нужно сделать на поиск общих интересов в процессе обсуждения конференции, обсудить понравившиеся доклады, понять заинтересован ли собеседник в диалоге. Для этого нужно слушать его, задавать различные вопросы и т.д. На первом этапе нельзя предлагать сотрудничество сразу, но можно обсуждать какие то области целиком.
Второй этап: Переход к обсуждению конкретной задачи. После завершения первого этапа, можно попробовать предложить человеку найти общие точки для сотрудничества. Обсудите какого рода работу вы можете сделать или предложить. Тут необходимо достичь соглашения в обсужджении какой-то конкретной работы, о которой вы договоритесь. Главное чтобы у вас слоижлось общее положительное мнение о будущей работе
Третий этап: Заключение договоренностей: на этом этапе необходимо получить уверенность в том, что все договоренности буду выполнены и обсуждение прошло не напрасно, а также обменяться контактами.

Начинается диалог с первого этапа, в случае перехода на другой, ты получишь соответствующую инструкцию в формате: "Нужно перейти на N этап диалога"
В самом начале необходимо поздороваться
Критерии ответных сообщений: Ответы должны быть сформулированы на простом русском разговорном языке.
 Ответы не должны быть длиннее 20 слов """


r=0.15

appr = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0]]
feelings = [[0, 0.5, 0.5, 0.6, 0.6], [0.6, 0.6, 0.6, 0.4, 0.4], [0.5, 0.5, 0.5, 0.6]]

schemes = [False, False, False]


#Текущее пространство - первое
prev_scheme = 1
current_scheme = 1
messages = [ {"role": "assistant", "content":start_promt} ]
while True:
    # Ввод сообщения
    message = input("User : ")
    #Оценка в текущем пространстве (этапе моральной схемы)
    action = intensional_calc(spaces[current_scheme-1], "gpt-3.5-turbo-0301", message) #получение оценок действия в текущем пространстве

    for i in range(len(appr[current_scheme-1])):
        appr[current_scheme-1][i] = (1 - r) * appr[current_scheme-1][i] + r * action[i] #перерасчет объективных оценок в текущем пространстве

    dist = euc_dist(appr[current_scheme-1], feelings[current_scheme-1]) #расстояния между векторами в текущем пространстве

    print(f"Оценки:{appr[current_scheme-1]}")
    print(f"Чувства:{feelings[current_scheme-1]}")
    print(f"Расстояние:{dist}")

    prev_scheme = current_scheme

    if dist >0.25: #если превышает пороговое значение
     #Тогда идет перерасчет чувств в текущем пространстве
        for i in range(len(appr[current_scheme-1])):
            feelings[current_scheme-1][i] = (1 - r) * feelings[current_scheme-1][i] + r * (appr[current_scheme-1][i]- feelings[current_scheme-1][i])
    else:
        #Иначе достигнута моральная схема
        schemes[current_scheme-1] = True
        #Переходим к следующей схеме
        current_scheme = min(current_scheme + 1, 3) #значение текущей, на которую совершили переход
    print(f'Моральные схемы: {schemes}')


    reply = answer_generate(message,messages, "gpt-3.5-turbo",spaces[current_scheme-1], feelings[current_scheme-1], prev_scheme, current_scheme)

    for i in range(len(appr[current_scheme-1])):
        appr[current_scheme-1][i] = (1 - r) * appr[current_scheme-1][i] + r * 0.3* feelings[current_scheme-1][i] #перерасчет объективных оценок в текущем пространстве посде фразы бота, т..к она тоже влияет на оценку человека, но в меньшей степени

    print(f"ChatGPT: {reply}")
    #print(f"Оценки:{appr}")
    #print(f"Чувства:{feelings}")
    #print(f"Оценки:{dist}")
    messages.append({"role": "user", "content": message})
    messages.append({"role": "assistant", "content": reply})



