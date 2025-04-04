import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

font_path = "C:/Windows/Fonts/THSarabun.ttf"  # ใช้ฟอนต์ที่เจอจากคำสั่งก่อนหน้า
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

# Set Sheet ID and Sheet Name
sheet_id = "1YX3HKrvKoYYQeSielEAUXf0DWAPm_by-BJP2lBpxpys"
sheet_name = "Dataproject"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

# Load data into Pandas
df = pd.read_csv(url)

# Select relevant columns
columns = ['อายุ : ','เวลาว่างเฉลี่ยต่อวันสำหรับพักผ่อน :','ระยะเวลาการเสพสื่อในแต่ละครั้ง :','ช่วงเวลาที่มักเสพสื่อมากที่สุด :', 'ประเภทของสื่อที่มักเลือกเสพในเวลาพักผ่อน :','ความรู้สึกและผลกระทบจากการเสพสื่อ', 'คุณคิดว่าการเสพสื่อของคุณมีผลกระทบต่อการใช้ชีวิตประจำวันหรือไม่?']
df_selected = df[columns].dropna()

# print("ข้อมูลที่เลือก:")
# print(df_selected.head())

# จัดลำดับช่วงอายุ
age_categories = ['ต่ำกว่า 18 ปี', '18-24 ปี', '25-34 ปี', '35-44 ปี', '45-54 ปี', '55-64 ปี']
df_selected['อายุ : '] = pd.Categorical(df_selected['อายุ : '], categories=age_categories, ordered=True)

# print("\nช่วงอายุที่เรียงลำดับแล้ว:")
# print(df_selected['อายุ : '].value_counts())

# # จัดลำดับเวลาว่างเฉลี่ยต่อวัน
# leisure_time_categories = ['น้อยกว่า 1 ชั่วโมง', '1-2 ชั่วโมง', '3-4 ชั่วโมง', 'มากกว่า 4 ชั่วโมง']
# df_selected['เวลาว่างเฉลี่ยต่อวันสำหรับพักผ่อน :'] = pd.Categorical(df_selected['เวลาว่างเฉลี่ยต่อวันสำหรับพักผ่อน :'], categories=leisure_time_categories, ordered=True)

# print("\nเวลาว่างเฉลี่ยต่อวันที่เรียงลำดับแล้ว:")
# print(df_selected['เวลาว่างเฉลี่ยต่อวันสำหรับพักผ่อน :'].value_counts())

# # จัดลำดับระยะเวลาการเสพสื่อ
# consumption_time_categories = ['น้อยกว่า 30 นาที', '30 นาที - 1 ชั่วโมง', '1 - 2 ชั่วโมง', 'มากกว่า 2 ชั่วโมง']
# df_selected['ระยะเวลาการเสพสื่อในแต่ละครั้ง :'] = pd.Categorical(df_selected['ระยะเวลาการเสพสื่อในแต่ละครั้ง :'], categories=consumption_time_categories, ordered=True)

# print("\nระยะเวลาการเสพสื่อที่เรียงลำดับแล้ว:")
# print(df_selected['ระยะเวลาการเสพสื่อในแต่ละครั้ง :'].value_counts())

# # จัดลำดับช่วงเวลาที่มักเสพสื่อมากที่สุด
# peak_time_categories = ['เช้า (06:00 - 09:00)', 'สาย (09:00 - 11:00)', 'กลางวัน (11:00 - 12:00)',
#                         'บ่าย (12:00 - 16:00)', 'เย็น (16:00 - 18:00)', 'ค่ำ (18:00 - 20:00)', 
#                         'ดึก (20:00 - 00:00)', 'ดึกหลังเที่ยงคืน (00:00 - 06:00)']
# df_selected['ช่วงเวลาที่มักเสพสื่อมากที่สุด :'] = pd.Categorical(df_selected['ช่วงเวลาที่มักเสพสื่อมากที่สุด :'], categories=peak_time_categories, ordered=True)

# print("\nช่วงเวลาที่เสพสื่อมากที่สุดที่เรียงลำดับแล้ว:")
# print(df_selected['ช่วงเวลาที่มักเสพสื่อมากที่สุด :'].value_counts())

# แยกประเภทสื่อเป็นคอลัมน์บูลีน
# media_types = ['โซเชียลมีเดีย', 'ดูหนัง/ซีรีส์', 'ฟังเพลง/พอดแคสต์', 'อ่านหนังสือ/บทความ', 'เล่นเกม']
# for media in media_types:
#     df_selected[media] = df_selected['ประเภทของสื่อที่มักเลือกเสพในเวลาพักผ่อน :'].apply(lambda x: media in str(x))

# print("\nตัวอย่างข้อมูลหลังแยกประเภทสื่อ:")
# print(df_selected.head())

# กำหนดคะแนนให้กับเวลาว่างเฉลี่ยต่อวัน
leisure_time_scores = {
    'น้อยกว่า 1 ชั่วโมง': 4,
    '1-2 ชั่วโมง': 3,
    '3-4 ชั่วโมง': 2,
    'มากกว่า 4 ชั่วโมง': 1
}

# แปลงข้อมูลในคอลัมน์ 'เวลาว่างเฉลี่ยต่อวัน' เป็นคะแนน
df_selected['คะแนนเวลาว่าง'] = df_selected['เวลาว่างเฉลี่ยต่อวันสำหรับพักผ่อน :'].map(leisure_time_scores)

print("คะแนนเวลาว่างเฉลี่ยต่อวัน:")
print(df_selected[['เวลาว่างเฉลี่ยต่อวันสำหรับพักผ่อน :', 'คะแนนเวลาว่าง']].head())

# กำหนดคะแนนให้กับระยะเวลาการเสพสื่อในแต่ละครั้ง
media_time_scores = {
    'น้อยกว่า 30 นาที': 1,
    '30 นาที - 1 ชั่วโมง': 2,
    '1 - 2 ชั่วโมง': 3,
    'มากกว่า 2 ชั่วโมง': 4
}

# แปลงข้อมูลในคอลัมน์ 'ระยะเวลาการเสพสื่อ' เป็นคะแนน
df_selected['คะแนนระยะเวลาการเสพสื่อ'] = df_selected['ระยะเวลาการเสพสื่อในแต่ละครั้ง :'].map(media_time_scores)

print("คะแนนระยะเวลาการเสพสื่อ:")
print(df_selected[['ระยะเวลาการเสพสื่อในแต่ละครั้ง :', 'คะแนนระยะเวลาการเสพสื่อ']].head())

# กำหนดคะแนนให้กับช่วงเวลาที่มักเสพสื่อมากที่สุด
peak_time_scores = {
    'เช้า (06:00 - 09:00)': 4,
    'สาย (09.00 - 11.00)': 4,
    'กลางวัน (11:00 - 12:00)': 4,
    'บ่าย (12:00 - 16:00)': 3,
    'เย็น (16:00 - 18:00)': 2,
    'ค่ำ (18.00 - 20.00)': 2,
    'ดึก (20.00 - 00.00)': 2,
    'ดึกหลังเที่ยงคืน (00.00 - 06.00)': 2
}

# แปลงข้อมูลในคอลัมน์ 'ช่วงเวลาที่เสพสื่อมากที่สุด' เป็นคะแนน
df_selected['คะแนนช่วงเวลาที่เสพสื่อ'] = df_selected['ช่วงเวลาที่มักเสพสื่อมากที่สุด :'].map(peak_time_scores)

print("คะแนนช่วงเวลาที่เสพสื่อมากที่สุด:")
print(df_selected[['ช่วงเวลาที่มักเสพสื่อมากที่สุด :', 'คะแนนช่วงเวลาที่เสพสื่อ']].head())


# กำหนดประเภทของสื่อที่เลือกเสพ
media_types = ['โซเชียลมีเดีย', 'ดูหนัง/ซีรีส์', 'ฟังเพลง/พอดแคสต์', 'อ่านหนังสือ/นิยาย/บทความออนไลน์/หนังสือพิมพ์', 'เล่นเกม']

# คำนวณคะแนนจากประเภทของสื่อที่เลือกเสพ
df_selected['คะแนนประเภทสื่อ'] = df_selected['ประเภทของสื่อที่มักเลือกเสพในเวลาพักผ่อน :'].apply(lambda x: sum(1 for media in media_types if media in str(x)))

print("คะแนนประเภทของสื่อที่เลือกเสพ:")
print(df_selected[['ประเภทของสื่อที่มักเลือกเสพในเวลาพักผ่อน :', 'คะแนนประเภทสื่อ']].head())

# กำหนดคะแนนให้กับความรู้สึกและผลกระทบจากการเสพสื่อ
impact_scores = {
    'ผ่อนคลาย/สบายใจ': 1,
    'สนุกสนาน/ตื่นเต้น': 1,
    'ได้รับความรู้ใหม่ๆ': 1,
    'เครียด/วิตกกังวล': -2,
    'เสียเวลาโดยไม่รู้ตัว': -1
}

# แปลงข้อมูลในคอลัมน์ 'ความรู้สึกและผลกระทบจากการเสพสื่อ' เป็นคะแนน
df_selected['คะแนนผลกระทบ'] = df_selected['ความรู้สึกและผลกระทบจากการเสพสื่อ'].apply(lambda x: sum(impact_scores.get(feeling, 0) for feeling in str(x).split(',')))

print("คะแนนความรู้สึกและผลกระทบจากการเสพสื่อ:")
print(df_selected[['ความรู้สึกและผลกระทบจากการเสพสื่อ', 'คะแนนผลกระทบ']].head())



# รวมคะแนนจากทุกคอลัมน์ที่เราสร้างไว้
df_selected['คะแนนรวม'] = df_selected[['คะแนนเวลาว่าง', 'คะแนนระยะเวลาการเสพสื่อ', 'คะแนนช่วงเวลาที่เสพสื่อ', 'คะแนนประเภทสื่อ', 'คะแนนผลกระทบ']].sum(axis=1)

print("คะแนนรวมของแต่ละคน:")
print(df_selected[['คะแนนเวลาว่าง', 'คะแนนระยะเวลาการเสพสื่อ', 'คะแนนช่วงเวลาที่เสพสื่อ', 'คะแนนประเภทสื่อ','คะแนนผลกระทบ', 'คะแนนรวม']].head())








# กำหนดสีตามช่วงอายุ
age_colors = {
    'ต่ำกว่า 18 ปี': 'skyblue',
    '18-24 ปี': 'blue',
    '25-34 ปี': 'lightgreen',
    '35-44 ปี': 'darkgreen',
    '45-54 ปี': '#D3A6F2',  # สีม่วงอ่อน
    '55-64 ปี': '#800080'  # สีม่วงเข้ม
}


# # สร้างคอลัมน์คะแนนความนิยมสื่อ (คำนวณจากคะแนนทั้งหมดที่เกี่ยวข้องกับสื่อ)
# df_selected['คะแนนความนิยมสื่อ'] = df_selected[['คะแนนประเภทสื่อ']].sum(axis=1)

# # สร้างกราฟ scatter โดยใช้ค่า "ผลกระทบจากการเสพสื่อ" เป็นแกน Y
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='คะแนนความนิยมสื่อ', y='คุณคิดว่าการเสพสื่อของคุณมีผลกระทบต่อการใช้ชีวิตประจำวันหรือไม่?', data=df_selected, hue='อายุ : ', palette=age_colors, s=100, marker='o')

# # เพิ่มชื่อกราฟและแกน
# plt.title('ความสัมพันธ์ระหว่างคะแนนความนิยมสื่อและผลกระทบจากการเสพสื่อ')
# plt.xlabel('คะแนนความนิยมสื่อ')
# plt.ylabel('ผลกระทบจากการเสพสื่อ')

# # แสดงกราฟ
# plt.legend(title='ช่วงอายุ')
# plt.show()

# สร้างกราฟ
plt.figure(figsize=(10, 6))

# กำหนดลำดับในแกน Y
impact_order = ['มีผลดี (เช่น ได้ความรู้, คลายเครียด)', 
                'ทั้งผลดีและผลเสีย', 
                'ไม่มีผลกระทบ', 
                'มีผลเสีย (เช่น ทำให้นอนดึก, ขาดสมาธิ)']

# ใช้ sns.stripplot โดยจัดลำดับแกน Y ตาม impact_order
sns.stripplot(x='คะแนนรวม', 
              y='คุณคิดว่าการเสพสื่อของคุณมีผลกระทบต่อการใช้ชีวิตประจำวันหรือไม่?', 
              data=df_selected, 
              jitter=True,  # ทำให้จุดไม่ทับกัน
              hue='อายุ : ',  # แยกตามช่วงอายุ
              palette=age_colors,  # ใช้สีที่กำหนดให้กับช่วงอายุ
              dodge=True,  # แยกกลุ่ม
              size=8,  # ขนาดของจุด
              marker='o',  # ใช้จุดวงกลม
              linewidth=0,  # ปิดการแสดงขอบสีเหลี่ยม
              order=impact_order)  # กำหนดลำดับที่ต้องการในแกน Y

# เพิ่มชื่อกราฟและแกน
plt.title('ความสัมพันธ์ระหว่างคะแนนรวมและผลกระทบจากการเสพสื่อ')
plt.xlabel('คะแนนรวม')
plt.ylabel('ผลกระทบจากการเสพสื่อ')

# แสดงกราฟ
plt.show()