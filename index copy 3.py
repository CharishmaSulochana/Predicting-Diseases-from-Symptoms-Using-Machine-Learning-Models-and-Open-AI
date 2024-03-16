from flask import Flask, render_template, request,jsonify
import pickle
import os
import sqlite3
import pickle
import numpy as np
from array import array
import importlib
import openai

openai.api_key = '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' 

currentdirectory=os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

pick_rf= pickle.load(open('final_rf_model.pkl', 'rb'))
pick_svm = pickle.load(open('final_svm_model.pkl', 'rb'))
pick_nb = pickle.load(open('final_nb_model.pkl', 'rb'))


def process_symptoms(symptoms_string, data_dict):
    symptoms_list = symptoms_string.split(",")
    input_data = [0] * len(data_dict["symptom_index"])

    for symptom in symptoms_list:
        index = data_dict["symptom_index"].get(symptom)
        if index is not None:
            input_data[index] = 1
        else:
            print(f"Symptom '{symptom}' not found in the index.")

    input_data = np.array(input_data).reshape(1, -1)
    return input_data

data_dict={'symptom_index': {'Umls:c0457096 Yellow sputum': 0,
  'Umls:c0425560 Cardiovascular finding': 1,
  'Umls:c0020440 Hypercapnia': 2,
  'Umls:c0581912 Heavy feeling': 3,
  'Umls:c0002416 Ambidexterity': 4,
  'Umls:c0521516 Polymyalgia': 5,
  'Umls:c0677500 Stinging sensation': 6,
  'Umls:c0392680 Shortness of breath': 7,
  'Umls:c0030252 Palpitation': 8,
  'Umls:c0020621 Hypokalemia': 9,
  'Umls:c0242453 Prostatism': 10,
  'Umls:c0948786 Blanch': 11,
  'Umls:c0085702 Monocytosis': 12,
  'Umls:c0237304 Noisy respiration': 13,
  'Umls:c0030232 Pallor': 14,
  'Umls:c0474505 Feces in rectum': 15,
  'Umls:c0037383 Sneeze': 16,
  'Umls:c0150041 Feeling hopeless': 17,
  'Umls:c0241235 Sputum purulent': 18,
  'Umls:c0038999 Swelling': 19,
  'Umls:c0238705 Left atrial hypertrophy': 20,
  'Umls:c0221232 Welt': 21,
  'Umls:c0232943 Intermenstrual heavy bleeding': 22,
  'Umls:c0013491 Ecchymosis': 23,
  'Umls:c0751466 Phonophobia': 24,
  'Umls:c0877040 Fear of falling': 25,
  'Umls:c0035508 Rhonchus': 26,
  'Umls:c0233467 Inappropriate affect': 27,
  'Umls:c0003126 Anosmia': 28,
  'Umls:c0879626 Adverse effect': 29,
  'Umls:c0239133 Hacking cough': 30,
  'Umls:c0311395 Lameness': 31,
  'Umls:c0241158 Scar tissue': 32,
  'Umls:c0043144 Wheezing': 33,
  'Umls:c0238844 Breath sounds decreased': 34,
  "Umls:c0018862 Heberden's node": 35,
  'Umls:c0020672 Hypothermia, natural': 36,
  'Umls:c0020578 Hyperventilation': 37,
  'Umls:c0233647 Neologism': 38,
  'Umls:c0232292 Chest tightness': 39,
  'Umls:c0741302 Atypia': 40,
  'Umls:c0542044 Incoherent': 41,
  'Umls:c0271202 Hemianopsia homonymous': 42,
  'Umls:c0026961 Mydriasis': 43,
  'Umls:c0009806 Constipation': 44,
  'Umls:c0015672 Fatigue': 45,
  'Umls:c0016512 Pain foot': 46,
  'Umls:c0436331 Symptom aggravating factors': 47,
  'Umls:c0241526 Unresponsiveness': 48,
  'Umls:c0744740 Heme positive': 49,
  'Umls:c0235231 Pin-point pupils': 50,
  'Umls:c0241252 Stool color yellow': 51,
  'Umls:c0541992 Groggy': 52,
  'Umls:c0871754 Frail': 53,
  'Umls:c0040822 Tremor': 54,
  'Umls:c0577559 Mass of body structure': 55,
  'Umls:c0456091 Large-for-dates fetus': 56,
  'Umls:c0858924 General discomfort': 57,
  'Umls:c0235129 Feeling strange': 58,
  'Umls:c0231441 Immobile': 59,
  'Umls:c0332575 Redness': 60,
  'Umls:c0231221 Asymptomatic': 61,
  'Umls:c0233481 Worry': 62,
  'Umls:c0152032 Urinary hesitation': 63,
  'Umls:c0232995 Gravida 0': 64,
  'Umls:c0424533 History of - blackout': 65,
  'Umls:c0424337 Hoard': 66,
  'Umls:c0748706 Side pain': 67,
  'Umls:c0744727 Hematocrit decreased': 68,
  'Umls:c0859032 Moan': 69,
  'Umls:c0847488 Unhappy': 70,
  'Umls:c0234979 Dysdiadochokinesia': 71,
  'Umls:c0581911 Heavy legs': 72,
  'Umls:c0041657 Unconscious state': 73,
  'Umls:c0018681 Headache': 74,
  'Umls:c0233565 Bradykinesia': 75,
  'Umls:c0522224 Paralyse': 76,
  'Umls:c0741453 Bedridden': 77,
  'Umls:c0019572 Hirsutism': 78,
  'Umls:c0392162 Clammy skin': 79,
  'Umls:c0000737 Pain abdominal': 80,
  'Umls:c0728899 Intoxication': 81,
  'Umls:c0264576 Mediastinal shift': 82,
  'Umls:c0018800 Cardiomegaly': 83,
  'Umls:c0043096 Decreased body weight': 84,
  'Umls:c1135120 Breakthrough pain': 85,
  'Umls:c1321756 Achalasia': 86,
  'Umls:c0423982 Rambling speech': 87,
  'Umls:c1456822 Claudication': 88,
  'Umls:c0425488 Rapid shallow breathing': 89,
  'Umls:c0085632 Indifferent mood': 90,
  'Umls:c0085631 Agitation': 91,
  'Umls:c0442739 No status change': 92,
  'Umls:c0013404 Dyspnea': 93,
  'Umls:c0221198 Lesion': 94,
  'Umls:c0264273 Nasal discharge present': 95,
  'Umls:c0220870 Lightheadedness': 96,
  'Umls:c0429091 R wave feature': 97,
  'Umls:c0018932 Hematochezia': 98,
  'Umls:c0231690 Titubation': 99,
  'Umls:c0023380 Lethargy': 100,
  'Umls:c0032739 ': 101,
  'Umls:c0556346 Alcohol binge episode': 102,
  'Umls:c0332601 Cushingoid facies': 103,
  'Umls:c0751495 Focal seizures': 104,
  'Umls:c0020625 Hyponatremia': 105,
  'Umls:c0016382 Flushing': 106,
  'Umls:c0240100 Jugular venous distention': 107,
  'Umls:c0425251 Bedridden': 108,
  'Umls:c0027066 Myoclonus': 109,
  'Umls:c0848621 Passed stones': 110,
  'Umls:c0239233 Satiety early': 111,
  'Umls:c0278141 Excruciating pain': 112,
  'Umls:c0008031 Pain chest': 113,
  'Umls:c0028084 Nightmare': 114,
  'Umls:c0086439 Hypokinesia': 115,
  'Umls:c0848168 Out of breath': 116,
  'Umls:c0013428 Dysuria': 117,
  'Umls:c0917801 Sleeplessness': 118,
  'Umls:c0558261 Terrify': 119,
  'Umls:c0232602 Retch': 120,
  'Umls:c1320716 Cardiovascular event': 121,
  'Umls:c0235396 Hypertonicity': 122,
  'Umls:c0027769 Nervousness': 123,
  'Umls:c0392701 Giddy mood': 124,
  'Umls:c0231530 Muscle twitch': 125,
  'Umls:c0549483 Abscess bacterial': 126,
  'Umls:c0277899 Pulse absent': 127,
  'Umls:c0007859 Pain neck': 128,
  'Umls:c0344315 Mood depressed': 129,
  'Umls:c1384489 Scratch marks': 130,
  'Umls:c0236018 Aura': 131,
  'Umls:c0232201 Sinus rhythm': 132,
  'Umls:c0541798 Awakening early': 133,
  'Umls:c0344232 Vision blurred': 134,
  'Umls:c1291692 Gravida 10': 135,
  'Umls:c0520886 St segment elevation': 136,
  'Umls:c0232498 Abdominal tenderness': 137,
  'Umls:c0277845 Retropulsion': 138,
  'Umls:c0857087 Dizzy spells': 139,
  'Umls:c0004093 Asthenia': 140,
  'Umls:c0231187 Decompensation': 141,
  'Umls:c0694547 Systolic ejection murmur': 142,
  'Umls:c0578150 Hemodynamically stable': 143,
  'Umls:c0015967 Fever': 144,
  'Umls:c0424530 Absences finding': 145,
  'Umls:c0277873 Nasal flaring': 146,
  'Umls:c0012833 Dizziness': 147,
  'Umls:c0000727 Abdomen acute': 148,
  'Umls:c0424230 Motor retardation': 149,
  'Umls:c0347938 Hypometabolism': 150,
  'Umls:c0020639 Hypoproteinemia': 151,
  'Umls:c0233844 Clumsiness': 152,
  'Umls:c0520888 T wave inverted': 153,
  'Umls:c0332573 Macule': 154,
  'Umls:c0019209 Hepatomegaly': 155,
  'Umls:c0917799 Hypersomnia': 156,
  'Umls:c0085636 Photophobia': 157,
  'Umls:c0034642 Rale': 158,
  'Umls:c0558141 Transsexual': 159,
  'Umls:c0850149 Non-productive cough': 160,
  'Umls:c0455769 Energy increased': 161,
  'Umls:c0262581 No known drug allergies': 162,
  'Umls:c0221151 Projectile vomiting': 163,
  'Umls:c0232258 Pansystolic murmur': 164,
  'Umls:c0557875 Tired': 165,
  'Umls:c0237154 Homelessness': 166,
  'Umls:c0277794 Extreme exhaustion': 167,
  'Umls:c0240962 Scleral icterus': 168,
  'Umls:c0231872 Egophony': 169,
  'Umls:c0497406 Overweight': 170,
  'Umls:c0006157 Breech presentation': 171,
  'Umls:c0848277 Room spinning': 172,
  'Umls:c0016927 Gag': 173,
  'Umls:c0856054 Mental status changes': 174,
  'Umls:c1517205 Flare': 175,
  'Umls:c0038990 Sweat': 176,
  'Umls:c1313921 Urinoma': 177,
  'Umls:c0008767 Cicatrisation': 178,
  'Umls:c0149746 Orthostasis': 179,
  'Umls:c0427108 General unsteadiness': 180,
  'Umls:c0474395 Behavior showing increased motor activity': 181,
  'Umls:c0019080 Haemorrhage': 182,
  'Umls:c0085619 Orthopnea': 183,
  'Umls:c0860096 Primigravida': 184,
  'Umls:c0577979 Frothy sputum': 185,
  'Umls:c0277797 Apyrexial': 186,
  'Umls:c0221470 Aphagia': 187,
  'Umls:c0235250 Hyperemesis': 188,
  'Umls:c0020303 Hydropneumothorax': 189,
  'Umls:c0028643 Numbness': 190,
  'Umls:c0018965 Hematuria': 191,
  'Umls:c0857256 Unwell': 192,
  'Umls:c0038002 Splenomegaly': 193,
  'Umls:c0239134 Productive cough': 194,
  'Umls:c0235710 Chest discomfort': 195,
  'Umls:c0020580 Hypesthesia': 196,
  'Umls:c0427008 Stiffness': 197,
  'Umls:c1444773 Throbbing sensation quality': 198,
  'Umls:c0424109 Weepiness': 199,
  'Umls:c0027497 Nausea': 200,
  'Umls:c0234233 Sore to touch': 201,
  'Umls:c0020461 Hyperkalemia': 202,
  'Umls:c0205400 Thicken': 203,
  'Umls:c0231890 Fremitus': 204,
  'Umls:c0002962 Angina pectoris': 205,
  'Umls:c0232766 Asterixis': 206,
  'Umls:c0028081 Night sweat': 207,
  'Umls:c0424092 Withdraw': 208,
  'Umls:c0009024 Clonus': 209,
  'Umls:c0557075 Has religious belief': 210,
  'Umls:c0151878 Qt interval prolonged': 211,
  'Umls:c0872410 Posturing': 212,
  'Umls:c0234518 Speech slurred': 213,
  'Umls:c0033774 Pruritus': 214,
  'Umls:c0020649 Hypotension': 215,
  'Umls:c0376405 Patient non compliance': 216,
  'Umls:c0231807 Dyspnea on exertion': 217,
  'Umls:c0037384 Snore': 218,
  'Umls:c0014394 Enuresis': 219,
  'Umls:c0234253 Rest pain': 220,
  'Umls:c0085606 Urgency of micturition': 221,
  'Umls:c0013362 Dysarthria': 222,
  'Umls:c0427055 Facial paresis': 223,
  'Umls:c0085639 Fall': 224,
  'Umls:c0020458 Hyperhidrosis disorder': 225,
  'Umls:c0042963 Vomiting': 226,
  'Umls:c0438696 Suicidal': 227,
  'Umls:c0085628 Stupor': 228,
  'Umls:c0019079 Haemoptysis': 229,
  'Umls:c0013132 Drool': 230,
  'Umls:c0270844 Tonic seizures': 231,
  'Umls:c0030552 Paresis': 232,
  'Umls:c0240805 Prodrome': 233,
  'Umls:c0337672 Nonsmoker': 234,
  'Umls:c1384606 Dyspareunia': 235,
  'Umls:c0000731 Distended abdomen': 236,
  'Umls:c0232854 Slowing of urinary stream': 237,
  'Umls:c0003962 Ascites': 238,
  'Umls:c0878661 Cushingoid habitus': 239,
  'Umls:c1291077 Abdominal bloating': 240,
  'Umls:c0232488 Colic abdominal': 241,
  'Umls:c0041834 Erythema': 242,
  'Umls:c0085602 Polydypsia': 243,
  'Umls:c0558195 Wheelchair bound': 244,
  'Umls:c0150045 Urge incontinence': 245,
  'Umls:c0702118 Abnormally hard consistency': 246,
  'Umls:c0554980 Moody': 247,
  'Umls:c0457097 Green sputum': 248,
  'Umls:c0010520 Cyanosis': 249,
  'Umls:c0085624 Burning sensation': 250,
  'Umls:c0242143 Uncoordination': 251,
  'Umls:c0424749 Feels hot/feverish': 252,
  'Umls:c0149696 Food intolerance': 253,
  'Umls:c0030318 Panic': 254,
  'Umls:c0424068 Verbal auditory hallucinations': 255,
  'Umls:c0020175 Hunger': 256,
  'Umls:c0281825 Disequilibrium': 257,
  'Umls:c0232997 Previous pregnancies 2': 258,
  'Umls:c0018834 Heartburn': 259,
  'Umls:c0027498 Nausea and vomiting': 260,
  'Umls:c0156543 Abortion': 261,
  'Umls:c0032781 Posterior rhinorrhea': 262,
  'Umls:c0003123 Anorexia': 263,
  'Umls:c0041667 Underweight': 264,
  'Umls:c0011991 Diarrhea': 265,
  'Umls:c1299586 Difficulty': 266,
  'Umls:c0278146 Shooting pain': 267,
  'Umls:c0424295 Behavior hyperactive': 268,
  'Umls:c0029053 Decreased translucency': 269,
  'Umls:c1096646 Transaminitis': 270,
  'Umls:c0234238 Ache': 271,
  'Umls:c0233492 Elation': 272,
  'Umls:c0205254 Sedentary': 273,
  'Umls:c0522336 Rolling of eyes': 274,
  'Umls:c0857516 Floppy': 275,
  'Umls:c0151706 Bleeding of vagina': 276,
  'Umls:c0233762 Hallucinations auditory': 277,
  'Umls:c0277823 Charleyhorse': 278,
  'Umls:c0006625 Cachexia': 279,
  'Umls:c0019825 Hoarseness': 280,
  'Umls:c0239301 Estrogen use': 281,
  'Umls:c0857199 Red blotches': 282,
  'Umls:c0232894 Pneumatouria': 283,
  'Umls:c0558089 Verbally abusive behavior': 284,
  'Umls:c0743482 Emphysematous change': 285,
  'Umls:c0700590 Sweating increased': 286,
  'Umls:c0004604 Pain back': 287,
  'Umls:c0020598 Hypocalcemia result': 288,
  'Umls:c0232726 Tenesmus': 289,
  'Umls:c0151315 Neck stiffness': 290,
  'Umls:c0233308 Spontaneous rupture of membranes': 291,
  'Umls:c0392699 Dysesthesia': 292,
  'Umls:c0008301 Choke': 293,
  'Umls:c0438716 Pressure chest': 294,
  'Umls:c0232257 Systolic murmur': 295,
  'Umls:c0455204 Homicidal thoughts': 296,
  'Umls:c0013144 Drowsiness': 297,
  'Umls:c0234544 Todd paralysis': 298,
  'Umls:c0038450 Stridor': 299,
  'Umls:c0558143 Macerated skin': 300,
  'Umls:c0424790 Rigor - temperature-associated observation': 301,
  'Umls:c0034880 Hyperacusis': 302,
  'Umls:c1319518 Underweight': 303,
  'Umls:c0037580 Soft tissue swelling': 304,
  'Umls:c0232267 Pericardial friction rub': 305,
  'Umls:c0700292 Hypoxemia': 306,
  'Umls:c0085593 Chill': 307,
  'Umls:c0239981 Hypoalbuminemia': 308,
  'Umls:c0234379 Tremor resting': 309,
  'Umls:c0004134 Ataxia': 310,
  'Umls:c0006318 Bruit': 311,
  'Umls:c0740880 Alcoholic withdrawal symptoms': 312,
  'Umls:c0239110 Consciousness clear': 313,
  'Umls:c1167754 Fecaluria': 314,
  'Umls:c0443260 Milky': 315,
  'Umls:c0003862 Arthralgia': 316,
  'Umls:c0744492 Guaiac positive': 317,
  'Umls:c0235198 Unable to concentrate': 318,
  'Umls:c0042571 Vertigo': 319,
  'Umls:c1305739 Presence of q wave': 320,
  'Umls:c0022107 Irritable mood': 321,
  'Umls:c0746619 Monoclonal': 322,
  'Umls:c0240233 Loose associations': 323,
  'Umls:c0234133 Extrapyramidal sign': 324,
  'Umls:c0520887 St segment depression': 325,
  'Umls:c0233763 Hallucinations visual': 326,
  'Umls:c0085635 Photopsia': 327,
  'Umls:c0241157 Pustule': 328,
  'Umls:c0036572 Seizure': 329,
  'Umls:c1260880 Snuffle': 330,
  'Umls:c0231218 Malaise': 331,
  'Umls:c0234215 Sensory discomfort': 332,
  'Umls:c0392674 Exhaustion': 333,
  'Umls:c0023222 Pain in lower limb': 334,
  'Umls:c0848340 Stuffy nose': 335,
  'Umls:c0028961 Oliguria': 336,
  'Umls:c0338656 Impaired cognition': 337,
  'Umls:c0559546 Adverse reaction': 338,
  "Umls:c0271276 Stahli's line": 339,
  'Umls:c0233070 Para 1': 340,
  'Umls:c0520966 Coordination abnormal': 341,
  'Umls:c0242429 Throat sore': 342,
  'Umls:c1511606 Cystic lesion': 343,
  'Umls:c1269955 Tumor cell invasion': 344,
  'Umls:c0018991 Hemiplegia': 345,
  'Umls:c0558361 Sniffle': 346,
  'Umls:c0429562 Superimposition': 347,
  'Umls:c0476273 Distress respiratory': 348,
  'Umls:c1273573 Unsteady gait': 349,
  'Umls:c0007398 Catatonia': 350,
  'Umls:c0428977 Bradycardia': 351,
  'Umls:c0232605 Regurgitates after swallowing': 352,
  'Umls:c0037763 Spasm': 353,
  'Umls:c0241705 Difficulty passing urine': 354,
  'Umls:c0423571 Abnormal sensation': 355,
  'Umls:c0036396 Sciatica': 356,
  'Umls:c0221166 Paraparesis': 357,
  'Umls:c0240813 Prostate tender': 358,
  'Umls:c0241938 Hypotonic': 359,
  'Umls:c0232517 Gurgle': 360,
  'Umls:c0231835 Tachypnea': 361,
  'Umls:c0553668 Labored breathing': 362,
  'Umls:c0016579 Formication': 363,
  'Umls:c0039070 Syncope': 364,
  'Umls:c0024103 Mass in breast': 365,
  'Umls:c0232695 Bowel sounds decreased': 366,
  'Umls:c0427629 Rhd positive': 367,
  'Umls:c0235634 Renal angle tenderness': 368,
  'Umls:c0742985 Debilitation': 369,
  'Umls:c0234866 Barking cough': 370,
  'Umls:c0234450 Sleepy': 371,
  'Umls:c0576456 Poor feeding': 372,
  'Umls:c1405524 Proteinemia': 373,
  'Umls:c0751229 Hypersomnolence': 374,
  'Umls:c0424000 Feeling suicidal': 375,
  'Umls:c0032617 Polyuria': 376,
  'Umls:c1513183 Metastatic lesion': 377,
  'Umls:c0034079 Lung nodule': 378,
  'Umls:c0040264 Tinnitus': 379,
  'Umls:c0542073 Lip smacking': 380,
  'Umls:c0541911 Dullness': 381,
  'Umls:c0019214 Hepatosplenomegaly': 382,
  'Umls:c0425449 Gasping for breath': 383,
  'Umls:c0578859 Disturbed family': 384,
  'Umls:c0278014 Decreased stool caliber': 385,
  'Umls:c0221150 Painful swallowing': 386,
  'Umls:c0149758 Poor dentition': 387,
  'Umls:c0030554 Paresthesia': 388,
  'Umls:c0024031 Low back pain': 389,
  'Umls:c0425491 Catching breath': 390,
  'Umls:c0232118 Pulsus paradoxus': 391,
  'Umls:c0231230 Fatigability': 392,
  'Umls:c0008033 Pleuritic pain': 393,
  'Umls:c0016204 Flatulence': 394,
  'Umls:c0231528 Myalgia': 395,
  'Umls:c0476287 Breath-holding spell': 396,
  'Umls:c0233071 Para 2': 397,
  'Umls:c0030193 Pain': 398,
  'Umls:c0239832 Numbness of hand': 399,
  "Umls:c0277977 Murphy's sign": 400,
  'Umls:c0740844 Air fluid level': 401,
  'Umls:c0026827 Muscle hypotonia': 402,
  'Umls:c0010200 Cough': 403,
  'Umls:c0043094 Weight gain': 404,
  'Umls:c0600142 Hot flush': 405,
  'Umls:c0312422 Blackout': 406},
 'predictions_classes': np.array(['UMLS:C0001175_acquired immuno-deficiency syndrome,UMLS:C0019682_HIV,UMLS:C0019693_hiv infections',
        'UMLS:C0001418_adenocarcinoma', 'UMLS:C0001511_adhesion',
        'UMLS:C0001973_chronic alcoholic intoxication',
        "UMLS:C0002395_Alzheimer's disease", 'UMLS:C0002871_anemia',
        'UMLS:C0002895_sickle cell anemia',
        'UMLS:C0003507_stenosis aortic valve', 'UMLS:C0003537_aphasia',
        'UMLS:C0003864_arthritis', 'UMLS:C0004096_asthma',
        'UMLS:C0004610_bacteremia',
        'UMLS:C0005001_benign prostatic hypertrophy',
        'UMLS:C0005586_bipolar disorder',
        'UMLS:C0006142_malignant neoplasm of breast^UMLS:C0678222_carcinoma breast',
        'UMLS:C0006266_spasm bronchial', 'UMLS:C0006277_bronchitis',
        'UMLS:C0006826_malignant neoplasms',
        'UMLS:C0006826_malignant neoplasms^UMLS:C1306459_primary malignant neoplasm',
        'UMLS:C0006840_candidiasis^UMLS:C0006849_oral candidiasis',
        'UMLS:C0007097_carcinoma',
        'UMLS:C0007102_malignant tumor of colon^UMLS:C0699790_carcinoma colon',
        'UMLS:C0007642_cellulitis',
        'UMLS:C0007787_transient ischemic attack',
        'UMLS:C0008325_cholecystitis',
        'UMLS:C0008350_cholelithiasis^UMLS:C0242216_biliary calculus',
        'UMLS:C0009319_colitis', 'UMLS:C0009676_confusion',
        'UMLS:C0010054_coronary arteriosclerosis^UMLS:C0010068_coronary heart disease',
        'UMLS:C0011127_decubitus ulcer',
        'UMLS:C0011168_deglutition disorder', 'UMLS:C0011175_dehydration',
        'UMLS:C0011206_delirium', 'UMLS:C0011253_delusion',
        'UMLS:C0011570_depression mental^UMLS:C0011581_depressive disorder',
        'UMLS:C0011847_diabetes', 'UMLS:C0011880_ketoacidosis diabetic ',
        'UMLS:C0012813_diverticulitis', 'UMLS:C0013405_paroxysmal dyspnea',
        'UMLS:C0014118_endocarditis', 'UMLS:C0014544_epilepsy',
        'UMLS:C0014549_tonic-clonic epilepsy^UMLS:C0494475_tonic-clonic seizures',
        'UMLS:C0015230_exanthema', 'UMLS:C0017152_gastritis',
        'UMLS:C0017160_gastroenteritis',
        'UMLS:C0017168_gastroesophageal reflux disease',
        'UMLS:C0017601_glaucoma', 'UMLS:C0018099_gout',
        'UMLS:C0018801_failure heart',
        'UMLS:C0018802_failure heart congestive',
        'UMLS:C0018989_hemiparesis', 'UMLS:C0019112_hemorrhoids',
        'UMLS:C0019158_hepatitis', 'UMLS:C0019163_hepatitis B',
        'UMLS:C0019196_hepatitis C',
        'UMLS:C0019204_primary carcinoma of the liver cells',
        'UMLS:C0019270_hernia', 'UMLS:C0019291_hernia hiatal',
        'UMLS:C0020433_hyperbilirubinemia',
        'UMLS:C0020443_hypercholesterolemia',
        'UMLS:C0020456_hyperglycemia', 'UMLS:C0020473_hyperlipidemia',
        'UMLS:C0020538_hypertensive disease',
        'UMLS:C0020542_hypertension pulmonary',
        'UMLS:C0020615_hypoglycemia', 'UMLS:C0020676_hypothyroidism',
        'UMLS:C0021167_incontinence', 'UMLS:C0021311_infection',
        'UMLS:C0021400_influenza', 'UMLS:C0022116_ischemia',
        'UMLS:C0022658_kidney disease',
        'UMLS:C0022660_kidney failure acute',
        'UMLS:C0022661_chronic kidney failure',
        'UMLS:C0023267_fibroid tumor',
        'UMLS:C0024117_chronic obstructive airway disease',
        'UMLS:C0024228_lymphatic diseases', 'UMLS:C0024299_lymphoma',
        'UMLS:C0024713_manic disorder', 'UMLS:C0025202_melanoma',
        'UMLS:C0026266_mitral valve insufficiency',
        'UMLS:C0027051_myocardial infarction',
        'UMLS:C0027627_neoplasm metastasis', 'UMLS:C0027651_neoplasm',
        'UMLS:C0027947_neutropenia', 'UMLS:C0028754_obesity',
        'UMLS:C0028756_obesity morbid',
        'UMLS:C0029408_degenerative polyarthritis',
        'UMLS:C0029443_osteomyelitis', 'UMLS:C0029456_osteoporosis',
        'UMLS:C0030305_pancreatitis', 'UMLS:C0030312_pancytopenia',
        'UMLS:C0030567_parkinson disease', 'UMLS:C0030920_ulcer peptic',
        'UMLS:C0031039_effusion pericardial^UMLS:C1253937_pericardial effusion body substance',
        'UMLS:C0031212_personality disorder', 'UMLS:C0032285_pneumonia',
        'UMLS:C0032290_pneumonia aspiration',
        'UMLS:C0032305_Pneumocystis carinii pneumonia',
        'UMLS:C0032326_pneumothorax', 'UMLS:C0033975_psychotic disorder',
        'UMLS:C0034063_edema pulmonary',
        'UMLS:C0034065_embolism pulmonary',
        'UMLS:C0034067_emphysema pulmonary',
        'UMLS:C0034186_pyelonephritis', 'UMLS:C0035078_failure kidney',
        'UMLS:C0036341_schizophrenia',
        'UMLS:C0036690_septicemia^UMLS:C0243026_systemic infection^UMLS:C1090821_sepsis (invertebrate)',
        'UMLS:C0038454_accident cerebrovascular',
        'UMLS:C0038663_suicide attempt', 'UMLS:C0039239_tachycardia sinus',
        'UMLS:C0040034_thrombocytopaenia',
        'UMLS:C0040961_tricuspid valve insufficiency',
        'UMLS:C0041912_upper respiratory infection',
        'UMLS:C0042029_infection urinary tract',
        'UMLS:C0085096_peripheral vascular disease',
        'UMLS:C0085584_encephalopathy', 'UMLS:C0087086_thrombus',
        'UMLS:C0149871_deep vein thrombosis',
        'UMLS:C0149931_migraine disorders', 'UMLS:C0233472_affect labile',
        'UMLS:C0242379_malignant neoplasm of lung^UMLS:C0684249_carcinoma of lung',
        'UMLS:C0376358_malignant neoplasm of prostate^UMLS:C0600139_carcinoma prostate',
        'UMLS:C0439857_dependence', 'UMLS:C0442874_neuropathy',
        'UMLS:C0497327_dementia', 'UMLS:C0546817_overload fluid',
        'UMLS:C0700613_anxiety state', 'UMLS:C0878544_cardiomyopathy',
        'UMLS:C1145670_respiratory failure', 'UMLS:C1258215_ileus',
        'UMLS:C1456784_paranoia', 'UMLS:C1510475_diverticulosis',
        'UMLS:C1565489_insufficiency renal', 'UMLS:C1623038_cirrhosis'],
       dtype=object)}


def get_disease_info(disease_name,prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=200
    )

    return response['choices'][0]['text'].strip()

conn = sqlite3.connect('symptoms_user.db')
cursor = conn.cursor()


# Function to store data in SQLite
def store_in_database(symptoms, rf_predict, svm_predict, nb_predict):
    conn = sqlite3.connect('symptoms_user.db')  
    cursor = conn.cursor()

    # Insert data into the Symptoms table
    cursor.execute('''
        INSERT INTO sym1 (symptoms, rf_predict, svm_predict, nb_predict)
        VALUES (?, ?, ?, ?)
    ''', (symptoms, rf_predict[14:], svm_predict[14:], nb_predict[14:]))

    conn.commit()
    conn.close()

@app.route('/')

def index():
    symptom_options_sorted = list(data_dict['symptom_index'].keys())  # Extract symptom options from data_dict
    symptom_options= sorted(symptom_options_sorted,key=lambda x: x[14].lower() if len(x) > 14 else '') 
    return render_template('index.html', symptom_options=symptom_options)


@app.route('/predict', methods=['POST'])
def predict():
    
    symptom1 = request.form['symptom1']
    symptom2 = request.form['symptom2']
    symptom3 = request.form['symptom3']
    symptom4 = request.form['symptom4']
    symptom5 = request.form['symptom5']
    symptom6 = request.form['symptom6']
    symptom7 = request.form['symptom7']
    symptom8 = request.form['symptom8']
    symptom9 = request.form['symptom9']
    symptom10 = request.form['symptom10']

    symptoms = ','.join([symptom1,symptom2,symptom3,symptom4,symptom5,symptom6,symptom7,symptom8,symptom9,symptom10])  # Concatenate selected symptoms

    input_data1=process_symptoms(symptoms,data_dict)
    rf_predict = data_dict["predictions_classes"][pick_rf.predict(input_data1)[0]]
    svm_predict = data_dict["predictions_classes"][pick_svm.predict(input_data1)[0]]
    nb_predict = data_dict["predictions_classes"][pick_nb.predict(input_data1)[0]]

    from collections import Counter
    predictions = [rf_predict, svm_predict, nb_predict]
    prediction_counts = Counter(predictions)
    most_common_prediction = prediction_counts.most_common(1)[0]
    most_common_disease = most_common_prediction[0]
    pred_disease=most_common_disease[14::]

    prompt_1=f"Retrieve information about {pred_disease}."
    prompt_2=f"How the {pred_disease} is caused and how does it get effected. "
    prompt_3=f"How does {pred_disease} impact different age groups?"
    prompt_4=f"How {pred_disease} to control it and recomend any home tips or medies to follow in points. Are there any lifestyle changes or dietary recommendations beneficial for managing {pred_disease}?"
    prompt_5=f"Which specialization doctor should  {pred_disease} effected people  meet .and in what severiety we should meet doctor. "
    prompt_6=f" How is {pred_disease} diagnosed?"
    prompt_7=f"What are the available treatment options for {pred_disease}?"
    prompt_8=f"Are there any specific risk factors associated with {pred_disease}?"

 
    p1=get_disease_info(pred_disease,prompt_1)
    p2=get_disease_info(pred_disease,prompt_2)
    p3=get_disease_info(pred_disease,prompt_3)
    p4=get_disease_info(pred_disease,prompt_4)
    p5=get_disease_info(pred_disease,prompt_5)
    p6=get_disease_info(pred_disease,prompt_6)
    p7=get_disease_info(pred_disease,prompt_7)
    p8=get_disease_info(pred_disease,prompt_8)

    store_in_database(symptoms, rf_predict, svm_predict, nb_predict)
    
    return render_template('result.html', symptoms=symptoms,rf_prediction=rf_predict[14::], svm_prediction=svm_predict[14::], nb_prediction=nb_predict[14::],p1=p1,p2=p2,p3=p3,p4=p4,p5=p5,p6=p6,p7=p7,p8=p8)


if __name__ == '__main__':
    app.run(debug=True)

