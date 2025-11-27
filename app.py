import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import urllib.parse  # For WhatsApp links

# ============================================================================
# KNOWLEDGE BASE V4: COMPLETE ASHA CURRICULUM (25 PROTOCOLS)
# ============================================================================
protocols_db = [
    # ------------------------------------------------------------------
    # SECTION 1: MATERNAL - PREGNANCY (ANTENATAL)
    # ------------------------------------------------------------------
    {
        "id": "mat_bleeding",
        "category": "Maternal",
        "topic_en": "Vaginal Bleeding (Pregnancy)",
        "topic_hi": "गर्भावस्था में रक्तस्राव (Bleeding)",
        "keywords": ["bleeding", "blood", "stain", "hemorrhage", "spotting", "period"],
        "content_en": "CRITICAL: Any bleeding in pregnancy is dangerous (APH).\n1. Do NOT do internal exam.\n2. Keep patient warm.\n3. Transport to FRU immediately.",
        "content_hi": "गंभीर: गर्भावस्था में रक्तस्राव खतरनाक है।\n1. आंतरिक जाँच न करें।\n2. मरीज को गर्म रखें।\n3. तुरंत अस्पताल ले जाएं।",
        "severity": "Critical",
        "action_msg": "URGENT: Pregnant woman with vaginal bleeding. Suspected Antepartum Hemorrhage. Moving to FRU."
    },
    {
        "id": "mat_vomit",
        "category": "Maternal",
        "topic_en": "Severe Vomiting (Hyperemesis)",
        "topic_hi": "अत्यधिक उल्टी (Severe Vomiting)",
        "keywords": ["vomit", "vomiting", "nausea", "sick", "throw up", "food", "morning sickness"],
        "content_en": "PROTOCOL: Hyperemesis.\n1. Can she keep fluids down?\n2. If NO (Dehydrated), Refer for IV fluids.\n3. Advise small, frequent meals (dry toast/biscuit).",
        "content_hi": "प्रोटोकॉल: अत्यधिक उल्टी।\n1. क्या वह पानी पी पा रही है?\n2. यदि नहीं (पानी की कमी), तो ड्रिप (IV) के लिए रेफर करें।\n3. थोड़ा-थोड़ा सूखा खाना (बिस्कुट) दें।",
        "severity": "High",
        "action_msg": "HIGH RISK: Pregnant woman with severe vomiting. Risk of dehydration."
    },
    {
        "id": "mat_bp",
        "category": "Maternal",
        "topic_en": "High BP / Pre-eclampsia",
        "topic_hi": "हाई ब्लड प्रेशर (High BP)",
        "keywords": ["bp", "headache", "vision", "blur", "dizzy", "swelling", "oedema", "face"],
        "content_en": "PROTOCOL: Pre-eclampsia Signs.\n1. Check BP. If > 140/90, it is severe.\n2. Check for urine albumin.\n3. Refer to Prevent Convulsions.",
        "content_hi": "प्रोटोकॉल: प्री-एक्लम्पसिया।\n1. बीपी जाँचें। यदि 140/90 से ज्यादा है, तो खतरा है।\n2. पेशाब की जाँच करें।\n3. दौरे रोकने के लिए तुरंत रेफर करें।",
        "severity": "High",
        "action_msg": "HIGH RISK: Pregnant woman with High BP (>140/90). Pre-eclampsia risk."
    },
    {
        "id": "mat_convulsions",
        "category": "Maternal",
        "topic_en": "Convulsions (Eclampsia)",
        "topic_hi": "दौरे पड़ना (Fits/Convulsions)",
        "keywords": ["convulsion", "fit", "seizure", "shake", "unconscious", "jerking", "teeth"],
        "content_en": "EMERGENCY: Eclampsia.\n1. Place patient on side.\n2. Do NOT put anything in mouth.\n3. Protect from injury.\n4. Transport immediately.",
        "content_hi": "आपातकालीन: दौरे (Eclampsia)।\n1. करवट लेकर लिटाएं।\n2. मुँह में चम्मच/उंगली न डालें।\n3. तुरंत जिला अस्पताल ले जाएं।",
        "severity": "Critical",
        "action_msg": "EMERGENCY: Pregnant woman having convulsions (Eclampsia). Transporting now."
    },
    {
        "id": "mat_anemia",
        "category": "Maternal",
        "topic_en": "Severe Anemia",
        "topic_hi": "गंभीर एनीमिया (खून की कमी)",
        "keywords": ["anemia", "pale", "weak", "tired", "hb", "hemoglobin", "white", "tongue"],
        "content_en": "ROUTINE: Anemia Management.\n1. If Hb < 7 g/dL, it is Severe -> Refer for Transfusion.\n2. If Hb 7-11, give IFA tablets twice daily.\n3. Eat green vegetables/jaggery.",
        "content_hi": "रूटीन: एनीमिया।\n1. यदि Hb 7 से कम है, तो खून चढ़ाना पड़ेगा -> रेफर करें।\n2. यदि 7-11 है, तो आयरन की गोली दें।\n3. हरी सब्जी और गुड़ खाने को कहें।",
        "severity": "Medium",
        "action_msg": "Referral: Woman with Severe Anemia (Hb < 7). Needs transfusion assessment."
    },

    # ------------------------------------------------------------------
    # SECTION 2: MATERNAL - LABOR & DELIVERY
    # ------------------------------------------------------------------
    {
        "id": "labor_water",
        "category": "Maternal",
        "topic_en": "Water Breaking (PROM)",
        "topic_hi": "पानी की थैली फटना (Water Break)",
        "keywords": ["water", "leak", "wet", "fluid", "break", "burst", "panties"],
        "content_en": "PROTOCOL: Premature Rupture of Membranes.\n1. If water breaks before labor pains -> Infection Risk.\n2. Do NOT wait at home.\n3. Refer for delivery within 24 hours.",
        "content_hi": "प्रोटोकॉल: पानी गिरना।\n1. दर्द से पहले पानी गिरना संक्रमण का खतरा है।\n2. घर पर इंतजार न करें।\n3. तुरंत अस्पताल जाएं।",
        "severity": "High",
        "action_msg": "URGENT: Water broke without labor pains. Risk of infection. Referring."
    },
    {
        "id": "labor_prolonged",
        "category": "Maternal",
        "topic_en": "Prolonged Labor (>12 Hours)",
        "topic_hi": "लंबे समय तक प्रसव पीड़ा",
        "keywords": ["long", "hours", "pain", "stuck", "pushing", "tired", "12 hours"],
        "content_en": "PROTOCOL: Prolonged Labor.\n1. If pains > 12 hours -> Risk of distress.\n2. If mother is exhausted or dehydrated -> Refer to FRU for C-Section assessment.",
        "content_hi": "प्रोटोकॉल: लंबी प्रसव पीड़ा।\n1. यदि दर्द 12 घंटे से ज्यादा हो -> खतरा।\n2. माँ थक गई है? -> बड़े अस्पताल (FRU) रेफर करें।",
        "severity": "High",
        "action_msg": "High Risk: Prolonged labor (>12 hours). Mother exhausted."
    },

    # ------------------------------------------------------------------
    # SECTION 3: MATERNAL - POST-PARTUM (AFTER BIRTH)
    # ------------------------------------------------------------------
    {
        "id": "pph_bleeding",
        "category": "Maternal",
        "topic_en": "Heavy Bleeding After Birth (PPH)",
        "topic_hi": "डिलीवरी के बाद ज्यादा खून (PPH)",
        "keywords": ["heavy", "soak", "pad", "clot", "flow", "birth", "delivery", "after"],
        "content_en": "EMERGENCY: Post-Partum Hemorrhage (PPH).\n1. Changing >2 pads in 30 mins?\n2. Massage the uterus (womb) to make it hard.\n3. Start breastfeeding immediately.\n4. Transport to Hospital.",
        "content_hi": "आपातकालीन: PPH (अधिक खून)।\n1. क्या 30 मिनट में 2 पैड भीग गए?\n2. पेट (गर्भाशय) की मालिश करें।\n3. तुरंत स्तनपान कराएं।\n4. एम्बुलेंस बुलाएं।",
        "severity": "Critical",
        "action_msg": "EMERGENCY: PPH detected. Heavy bleeding after delivery. Massaging uterus and transporting."
    },
    {
        "id": "pp_sepsis",
        "category": "Maternal",
        "topic_en": "Fever After Delivery (Sepsis)",
        "topic_hi": "डिलीवरी के बाद बुखार (Sepsis)",
        "keywords": ["fever", "smell", "discharge", "pus", "stink", "pain", "stomach"],
        "content_en": "PROTOCOL: Puerperal Sepsis.\n1. High fever + foul smelling discharge?\n2. Lower abdominal pain?\nACTION: Needs Antibiotics. Refer immediately.",
        "content_hi": "प्रोटोकॉल: प्रसव के बाद संक्रमण।\n1. तेज बुखार और बदबूदार पानी?\n2. पेट के निचले हिस्से में दर्द?\nकार्रवाई: एंटीबायोटिक की जरूरत है। रेफर करें।",
        "severity": "High",
        "action_msg": "High Risk: Mother has fever and foul discharge post-delivery. Suspected Sepsis."
    },
    {
        "id": "breast_mastitis",
        "category": "Maternal",
        "topic_en": "Breast Pain / Mastitis",
        "topic_hi": "स्तन में दर्द / सूजन (Mastitis)",
        "keywords": ["breast", "nipple", "pain", "crack", "hard", "lump", "milk", "feed"],
        "content_en": "COUNSELING: Breast Problems.\n1. Hard/Red lump = Mastitis.\n2. Keep feeding from that breast to empty it.\n3. Apply warm cloth compress.\n4. If fever -> Refer for antibiotics.",
        "content_hi": "सलाह: स्तन में गांठ/दर्द।\n1. लाल/कड़ी गांठ = मैस्टाइटिस।\n2. दूध पिलाना बंद न करें (गांठ खाली करें)।\n3. गर्म कपड़े से सेकें।\n4. बुखार हो तो डॉक्टर को दिखाएं।",
        "severity": "Medium",
        "action_msg": "Consultation: Mother reporting breast pain/lump. Advised warm compress and continued feeding."
    },

    # ------------------------------------------------------------------
    # SECTION 4: CHILD - NEWBORN (0-28 DAYS)
    # ------------------------------------------------------------------
    {
        "id": "nb_sepsis",
        "category": "Child",
        "topic_en": "Newborn Sepsis (Infection)",
        "topic_hi": "नवजात शिशु में संक्रमण (Sepsis)",
        "keywords": ["fever", "cold", "feed", "suck", "milk", "lethargic", "cry", "weak"],
        "content_en": "CRITICAL: Newborn Danger Signs.\n1. Stopped feeding?\n2. Cold to touch OR High Fever?\n3. Lethargic (No movement)?\n4. Chest indrawing?\nACTION: Immediate Referral to SNCU.",
        "content_hi": "गंभीर: नवजात खतरे के संकेत।\n1. दूध नहीं पी रहा?\n2. शरीर ठंडा या तेज बुखार?\n3. सुस्त है?\n4. पसली चल रही है?\nकार्रवाई: तुरंत SNCU रेफर करें।",
        "severity": "Critical",
        "action_msg": "URGENT: Newborn with danger signs (No feed/Fever). Suspected Sepsis. Moving to SNCU."
    },
    {
        "id": "nb_jaundice",
        "category": "Child",
        "topic_en": "Newborn Jaundice",
        "topic_hi": "पीलिया (Jaundice)",
        "keywords": ["yellow", "skin", "eyes", "palm", "sole", "feet", "jaundice"],
        "content_en": "PROTOCOL: Jaundice.\n1. Yellow palms/soles is DANGER.\n2. Appears within 24 hours of birth -> Danger.\nACTION: Refer for Phototherapy.",
        "content_hi": "प्रोटोकॉल: पीलिया।\n1. हथेली/तलवे पीले होना खतरे की बात है।\n2. जन्म के 24 घंटे के अंदर पीलिया -> गंभीर।\nकार्रवाई: फोटोथेरेपी के लिए भेजें।",
        "severity": "High",
        "action_msg": "Referral: Newborn with severe jaundice (Palms/Soles yellow)."
    },
    {
        "id": "nb_lbw",
        "category": "Child",
        "topic_en": "Low Birth Weight / KMC",
        "topic_hi": "कम वजन का बच्चा (KMC)",
        "keywords": ["small", "weight", "tiny", "warm", "kmc", "kangaroo", "kg"],
        "content_en": "CARE: Kangaroo Mother Care (KMC).\n1. If weight < 2.5 kg.\n2. Keep baby skin-to-skin on mother's chest 24x7.\n3. Feed every 2 hours.",
        "content_hi": "देखभाल: कंगारू मदर केयर।\n1. यदि वजन 2.5 किलो से कम है।\n2. बच्चे को माँ की छाती से चिपका कर रखें।\n3. हर 2 घंटे में दूध पिलाएं।",
        "severity": "Medium",
        "action_msg": "Counseling: Low Birth Weight baby. KMC technique demonstrated."
    },
    {
        "id": "nb_cord",
        "category": "Child",
        "topic_en": "Umbilical Cord Care",
        "topic_hi": "नाभि की देखभाल (Cord Care)",
        "keywords": ["cord", "navel", "stump", "red", "pus", "blood", "infection"],
        "content_en": "ROUTINE: Cord Care.\n1. Keep cord dry and clean.\n2. Do NOT apply cow dung/oil/powder.\n3. If red/pus -> Infection (Refer).",
        "content_hi": "रूटीन: नाभि की देखभाल।\n1. नाभि को सूखा रखें।\n2. गोबर, तेल या पाउडर न लगाएं।\n3. यदि लाल हो या मवाद आए -> संक्रमण है (रेफर करें)।",
        "severity": "Low",
        "action_msg": "Counseling: Umbilical cord infection check. Referred if pus found."
    },

    # ------------------------------------------------------------------
    # SECTION 5: CHILD - ILLNESS (> 1 MONTH)
    # ------------------------------------------------------------------
    {
        "id": "child_pneumonia",
        "category": "Child",
        "topic_en": "Pneumonia (Fast Breathing)",
        "topic_hi": "निमोनिया (तेज सांस)",
        "keywords": ["breath", "cough", "fast", "chest", "ribs", "breathing", "cold", "pasli"],
        "content_en": "PROTOCOL: Pneumonia Check.\n1. Count breaths per minute (Age < 2mo: >60 | 2-12mo: >50).\n2. Look for Chest Indrawing.\nACTION: Give Amoxicillin & Refer.",
        "content_hi": "प्रोटोकॉल: निमोनिया।\n1. सांस की गति गिनें।\n2. क्या पसली चल रही है?\nकार्रवाई: एमोक्सिसिलिन (Amoxicillin) दें और रेफर करें।",
        "severity": "High",
        "action_msg": "HIGH RISK: Child with fast breathing/chest indrawing. Suspected Pneumonia."
    },
    {
        "id": "child_diarrhea",
        "category": "Child",
        "topic_en": "Diarrhea & Dehydration",
        "topic_hi": "दस्त और पानी की कमी",
        "keywords": ["loose", "motion", "stool", "watery", "diarrhea", "vomit", "thirsty", "pinch"],
        "content_en": "PROTOCOL: Diarrhea.\n1. Assess Dehydration: Sunken eyes? Skin pinch goes back slow?\n2. Give ORS + Zinc (14 days).\n3. DANGER: Blood in stool (Dysentery) -> Refer.",
        "content_hi": "प्रोटोकॉल: दस्त।\n1. पानी की कमी: धंसी आँखें? त्वचा धीरे वापस जाती है?\n2. ORS और जिंक (Zinc) दें।\n3. खतरा: लैट्रिन में खून? -> रेफर करें।",
        "severity": "Medium",
        "action_msg": "Follow-up: Child with diarrhea. ORS/Zinc prescribed. Dehydration checked."
    },
    {
        "id": "child_malnutrition",
        "category": "Child",
        "topic_en": "Severe Malnutrition (SAM)",
        "topic_hi": "गंभीर कुपोषण (SAM)",
        "keywords": ["thin", "weak", "weight", "swollen", "legs", "muac", "tape", "eat"],
        "content_en": "PROTOCOL: SAM (Severe Acute Malnutrition).\n1. MUAC Tape < 11.5 cm (Red Zone).\n2. Swelling in both feet (Oedema).\nACTION: Refer to Nutrition Rehab Centre (NRC).",
        "content_hi": "प्रोटोकॉल: गंभीर कुपोषण।\n1. फीता (MUAC) < 11.5 सेमी (लाल रंग)।\n2. दोनों पैरों में सूजन।\nकार्रवाई: पोषण केंद्र (NRC) भेजें।",
        "severity": "High",
        "action_msg": "Referral: Child identified as SAM (Red Zone MUAC). Referred to NRC."
    },
    {
        "id": "child_malaria",
        "category": "Child",
        "topic_en": "Fever / Malaria",
        "topic_hi": "बुखार / मलेरिया",
        "keywords": ["fever", "hot", "shiver", "cold", "mosquito", "malaria", "temperature"],
        "content_en": "PROTOCOL: Fever.\n1. Fever with chills/rigors?\n2. Perform RDT (Rapid Diagnostic Test) for Malaria.\n3. Give Paracetamol for fever control.",
        "content_hi": "प्रोटोकॉल: बुखार।\n1. क्या ठंड लगकर बुखार है?\n2. मलेरिया की जाँच (RDT) किट से करें।\n3. बुखार के लिए पैरासिटामोल दें।",
        "severity": "Medium",
        "action_msg": "Action: Child with fever. Malaria RDT recommended."
    },

    # ------------------------------------------------------------------
    # SECTION 6: GENERAL / FAMILY PLANNING
    # ------------------------------------------------------------------
    {
        "id": "fp_spacing",
        "category": "Maternal",
        "topic_en": "Family Planning (Spacing)",
        "topic_hi": "परिवार नियोजन (अंतर रखना)",
        "keywords": ["birth control", "gap", "space", "pill", "copper-t", "mala-n", "chhaya"],
        "content_en": "COUNSELING: Spacing Methods.\n1. IUCD (Copper-T): Effective for 5-10 years.\n2. Mala-N: Daily pills.\n3. Chhaya: Weekly pills (Non-hormonal).\n4. Condoms: Safe & easy.",
        "content_hi": "सलाह: बच्चों में अंतर।\n1. कॉपर-टी (IUCD): 5-10 साल के लिए।\n2. माला-एन: रोज की गोली।\n3. छाया: हफ्ते की गोली।\n4. निरोध (Condoms): सुरक्षित।",
        "severity": "Low",
        "action_msg": "Counseling: Family planning options (Spacing) explained."
    },
    {
        "id": "fp_limiting",
        "category": "Maternal",
        "topic_en": "Family Planning (Operation)",
        "topic_hi": "नसबंदी (Operation)",
        "keywords": ["operation", "sterilization", "tubectomy", "vasectomy", "stop", "limit"],
        "content_en": "COUNSELING: Permanent Methods.\n1. Tubectomy: Female sterilization.\n2. NSV: Male sterilization (No stitch, 10 mins).\n3. Incentive available from Govt.",
        "content_hi": "सलाह: नसबंदी।\n1. महिला नसबंदी (Tubectomy)।\n2. पुरुष नसबंदी (NSV): बिना टांका, 10 मिनट।\n3. सरकारी प्रोत्साहन राशि उपलब्ध है।",
        "severity": "Low",
        "action_msg": "Referral: Couple interested in permanent sterilization. Referred to CHC."
    },
    {
        "id": "mat_diagnosis",
        "category": "Maternal",
        "topic_en": "Pregnancy Test (Nischay Kit)",
        "topic_hi": "गर्भावस्था की जाँच (निश्चय किट)",
        "keywords": ["test", "kit", "check", "urine", "period", "missed", "nischay"],
        "content_en": "PROTOCOL: Nischay Kit Test.\n1. Use morning urine.\n2. Put 2 drops in the sample well.\n3. Wait 5 mins.\nRESULT:\n- 2 Lines = Pregnant.\n- 1 Line = Not Pregnant.",
        "content_hi": "प्रोटोकॉल: निश्चय किट।\n1. सुबह के पेशाब का उपयोग करें।\n2. 2 बूंदें डालें।\n3. 5 मिनट रुकें।\nपरिणाम:\n- 2 लाइन = गर्भवती है।\n- 1 लाइन = गर्भवती नहीं है।",
        "severity": "Low",
        "action_msg": "Counseling: Pregnancy test guidance provided using Nischay Kit."
    },
    {
        "id": "mat_depression",
        "category": "Maternal",
        "topic_en": "Post-Partum Sadness/Mood",
        "topic_hi": "प्रसव के बाद उदासी (Mood Changes)",
        "keywords": ["sad", "cry", "mood", "unhappy", "angry", "depression", "feeling"],
        "content_en": "COUNSELING: Post-Partum Mood Changes.\n1. It is common to feel sad/weepy after delivery.\n2. Needs family support and rest.\n3. If severe or talks of harm -> REFER immediately.",
        "content_hi": "सलाह: प्रसव के बाद उदासी।\n1. डिलीवरी के बाद रोना या उदास होना आम है।\n2. परिवार के सहयोग की जरूरत है।\n3. यदि माँ खुद को चोट पहुँचाने की बात करे -> तुरंत डॉक्टर को दिखाएं।",
        "severity": "Medium",
        "action_msg": "Referral: Mother showing signs of severe post-partum depression."
    },
    {
        "id": "nb_hypothermia",
        "category": "Child",
        "topic_en": "Baby feels Cold (Hypothermia)",
        "topic_hi": "बच्चा ठंडा पड़ गया है (Hypothermia)",
        "keywords": ["cold", "feet", "blue", "warm", "temperature", "winter", "shiver"],
        "content_en": "PROTOCOL: Hypothermia (Cold Stress).\n1. Feel the feet. If cold -> Baby is in danger.\n2. Skin-to-Skin contact (Kangaroo Care) immediately.\n3. Cover head with cap.\n4. Warm the room.",
        "content_hi": "प्रोटोकॉल: ठंडा बुखार (Hypothermia)।\n1. पैर छुएं। यदि ठंडे हैं -> खतरा है।\n2. माँ की छाती से चिपका कर रखें (KMC)।\n3. टोपी पहनाएं।\n4. कमरा गर्म रखें।",
        "severity": "Critical",
        "action_msg": "URGENT: Baby is hypothermic (Cold to touch). Rewarming initiated. Transporting if no improvement."
    },
    {
        "id": "nb_eyes",
        "category": "Child",
        "topic_en": "Sticky Eyes / Pus",
        "topic_hi": "आँख से मवाद आना (Sticky Eyes)",
        "keywords": ["eye", "pus", "sticky", "discharge", "water", "yellow", "red"],
        "content_en": "PROTOCOL: Eye Care.\n1. Clean eyes with sterile water and cotton.\n2. Apply Tetracycline eye ointment.\n3. If swelling/redness persists -> Refer.",
        "content_hi": "प्रोटोकॉल: आँखों की देखभाल।\n1. आँखों को साफ पानी और रूई से पोंछें।\n2. टेट्रासाइक्लिन (Tetracycline) मलहम लगाएं।\n3. यदि सूजन है -> डॉक्टर को दिखाएं।",
        "severity": "Medium",
        "action_msg": "Action: Newborn eye infection suspected. Cleaning and ointment advised."
    },
    {
        "id": "anc_schedule",
        "category": "Maternal",
        "topic_en": "ANC Schedule (Check-up Dates)",
        "topic_hi": "जाँच की तारीखें (ANC Schedule)",
        "keywords": ["when", "visit", "checkup", "schedule", "date", "anc", "doctor", "time"],
        "content_en": "PROTOCOL: 4 Mandatory Visits.\n1. 1st Visit: Within 12 weeks (Registration).\n2. 2nd Visit: 14-26 weeks.\n3. 3rd Visit: 28-34 weeks.\n4. 4th Visit: After 36 weeks.",
        "content_hi": "प्रोटोकॉल: 4 जरूरी जाँचें।\n1. पहली: 3 महीने के अंदर (पंजीकरण)।\n2. दूसरी: चौथा-छठा महीना।\n3. तीसरी: सातवां-आठवां महीना।\n4. चौथी: नौवें महीने में।",
        "severity": "Low",
        "action_msg": "Counseling: ANC Check-up schedule explained."
    }
]
# ============================================================================
# 2. LOGIC CORE: PARALLEL TRIAGE ENGINE
# ============================================================================


class TriageEngine:
    def __init__(self, db):
        self.db = db
        self.model = None
        self.initialized = False

    @st.cache_resource
    def load_model(_self):
        return SentenceTransformer('all-MiniLM-L6-v2')

    def initialize(self):
        if not self.initialized:
            self.model = self.load_model()
            # Pre-compute embeddings
            for p in self.db:
                # Embed English content + Keywords for better matching
                text_to_embed = f"{p['topic_en']} {p['content_en']} {' '.join(p['keywords'])}"
                p['embedding'] = self.model.encode(text_to_embed)
            self.initialized = True

    def detect_scope(self, text):
        """ The Firewall: Decides if a query is Maternal or Child """
        text = text.lower()
        child_triggers = ['baby', 'child', 'infant',
                          'newborn', 'kid', 'boy', 'girl']
        mat_triggers = ['mother', 'mom', 'woman',
                        'pregnant', 'lady', 'she', 'her', 'wife']

        if any(w in text for w in child_triggers):
            return "Child"
        if any(w in text for w in mat_triggers):
            return "Maternal"
        return "General"

    def search_single(self, query, scope_filter="General"):
        """ Runs a search against a specific scope """
        self.initialize()
        query_vec = self.model.encode(query)

        best_score = -1
        best_doc = None

        for doc in self.db:
            # Firewall Rule: Skip if scope doesn't match
            if scope_filter != "General" and doc['category'] != scope_filter:
                continue

            # Similarity Calculation
            score = np.dot(doc['embedding'], query_vec)
            if score > best_score:
                best_score = score
                best_doc = doc

        # Threshold (Noise filter)
        if best_score < 0.25:
            return None
        return best_doc

    def process_query(self, raw_input):
        """ The Splitter: Handles Multi-Intent Logic """
        results = []

        # 1. SPLIT: Simple keyword splitter for "and" / ","
        # In a real app, use an LLM for this. Here we use a heuristic.
        sub_queries = [q.strip()
                       for q in raw_input.replace(',', ' and ').split(' and ')]

        unique_ids = set()

        for sub_q in sub_queries:
            if len(sub_q) < 3:
                continue

            # 2. DETECT SCOPE
            scope = self.detect_scope(sub_q)

            # 3. RETRIEVE
            match = self.search_single(sub_q, scope)

            if match and match['id'] not in unique_ids:
                results.append({
                    "query_segment": sub_q,
                    "scope_detected": scope,
                    "protocol": match
                })
                unique_ids.add(match['id'])

        return results

# ============================================================================
# 3. UI LAYER: CLINICAL DASHBOARD
# ============================================================================


def main():
    st.set_page_config(page_title="ASHA AI Assistant",
                       page_icon="🏥", layout="wide")

    # Initialize Engine
    if 'engine' not in st.session_state:
        st.session_state.engine = TriageEngine(protocols_db)

    # --- HEADER ---
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title("🏥 ASHA Health Assistant")
        st.caption("AI Decision Support for Rural Health Workers")
    with col_h2:
        # LANGUAGE TOGGLE
        is_hindi = st.toggle("हिंदी (Hindi Mode)", value=False)

    st.divider()

    # --- INPUT SECTION (Touch Interface) ---
    st.markdown("### 📝 Patient Symptoms")

    # Quick Chips
    chips_col1, chips_col2 = st.columns(2)
    with chips_col1:
        st.markdown("**Mother (Maternal):**")
        c1, c2, c3 = st.columns(3)
        if c1.button("🩸 Bleeding"):
            st.session_state.q_input = "Mother has vaginal bleeding"
        if c2.button("🤕 Headache"):
            st.session_state.q_input = "Mother has severe headache"
        if c3.button("⚪ Pale/Weak"):
            st.session_state.q_input = "Mother looks pale and tired"

    with chips_col2:
        st.markdown("**Child (Newborn):**")
        c4, c5, c6 = st.columns(3)
        if c4.button("🤒 Fever"):
            st.session_state.q_input = "Baby has high fever"
        if c5.button("🍼 No Feed"):
            st.session_state.q_input = "Baby stopped feeding"
        if c6.button("🫁 Fast Breath"):
            st.session_state.q_input = "Child has fast breathing"

    # Search Bar
    query = st.text_input("Or type description:", value=st.session_state.get(
        'q_input', ''), placeholder="Ex: Mother has high BP and Baby has fever")

    # --- PROCESS & DISPLAY ---
    if query:
        with st.spinner("Analyzing Clinical Protocols..."):
            results = st.session_state.engine.process_query(query)

        if not results:
            st.warning(
                "No specific protocol found. Please consult Medical Officer.")
        else:
            # CHECK FOR COMBINED RISK (Reasoning Layer)
            critical_count = sum(
                1 for r in results if r['protocol']['severity'] == 'Critical')

            if critical_count > 0:
                st.error(
                    f"🚨 CRITICAL ALERT: {critical_count} DANGER SIGNS DETECTED. IMMEDIATE REFERRAL REQUIRED.")

            # Display Cards Side-by-Side
            cols = st.columns(len(results))

            for idx, res in enumerate(results):
                proto = res['protocol']

                with cols[idx]:
                    # Severity Badge
                    color_map = {"Critical": "red",
                                 "High": "orange", "Medium": "blue"}
                    st.markdown(
                        f":{color_map.get(proto['severity'], 'grey')}[**{proto['severity'].upper()}**]")

                    # Content (Bilingual)
                    topic = proto['topic_hi'] if is_hindi else proto['topic_en']
                    content = proto['content_hi'] if is_hindi else proto['content_en']

                    st.subheader(topic)
                    st.info(content)

                    # Debug Context Info
                    st.caption(
                        f"Context: {res['scope_detected']} | Trigger: '{res['query_segment']}'")

                    # WhatsApp Action Button
                    if proto['severity'] in ['Critical', 'High']:
                        wa_text = urllib.parse.quote(proto['action_msg'])
                        st.link_button(
                            "📲 Refer on WhatsApp",
                            f"https://wa.me/?text={wa_text}",
                            type="primary"
                        )


if __name__ == "__main__":
    main()
