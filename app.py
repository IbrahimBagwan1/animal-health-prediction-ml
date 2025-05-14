import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import pickle

# Page configuration
st.set_page_config(
    page_title="Animal Symptom Predictor",
    page_icon="🦠",
    layout="wide"
)

# Custom CSS for a clean, modern look
st.markdown(
    """
    <style>
    /* Backgrounds */
    .reportview-container { background-color: #f7f9fc; }
    .css-1d391kg { background-color: #1f2833; padding: 2rem; }
    /* Sidebar header */
    .css-1d391kg h2 { color: #66fcf1; }
    /* Selectboxes */
    .stSelectbox>div>div { height: 3rem; border-radius: 8px; border: 1px solid #ccc; padding-left: 0.5rem; }
    /* Buttons */
    .stButton>button { width: 100%; padding: 0.75rem; border-radius: 8px; background-color: #45a29e; color: #ffffff; font-size: 1rem; }
    /* Metric */
    .stMetric-value { font-size: 2rem; color: #c5c6c7; }
    </style>
    """, unsafe_allow_html=True
)

# ----- Data & Model Loading -----
@st.cache_data
def load_data(path='data.csv'):
    df = pd.read_csv(path)
    animals = df['AnimalName'].unique().tolist()
    symptom_cols = ['symptoms1','symptoms2','symptoms3','symptoms4','symptoms5']
    symptoms = pd.unique(df[symptom_cols].values.ravel()).tolist()
    return animals, symptoms

@st.cache_resource
def load_model(path='random_forest_model.joblib'):
    return load(path)

@st.cache_data
def load_encoder(path='onehot_encoder.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

animals, symptoms = load_data()
model = load_model()
encoder = load_encoder()

# ----- Sidebar Inputs -----
with st.sidebar:
    st.header("Input Parameters")
    animal = st.selectbox(
        "Animal Name", options=[None] + animals, index=0,
        format_func=lambda x: "Select an animal" if x is None else x
    )
    # symptom pickers
    picks = []
    for i in range(1, 6):
        picks.append(
            st.selectbox(
                f"Symptom {i}", options=[None] + symptoms, index=0,
                format_func=lambda x, i=i: f"Select symptom {i}" if x is None else x
            )
        )
    predict_btn = st.button("Predict")

# ----- Main Page -----
st.title("🦠 Animal Symptom Danger Predictor")
st.write("Select an animal and at least three symptoms in the sidebar, then click **Predict**.")

if predict_btn:
    # Validate: at least 3 symptoms and animal selected
    selected_symptoms = [s for s in picks if s is not None]
    if animal is None or len(selected_symptoms) < 3:
        st.error("Please select an animal and at least three symptoms before predicting.")
    else:
        # Fill None with placeholder for encoder (will be ignored)
        input_row = [animal] + [s if s is not None else "" for s in picks]
        cols = ['AnimalName','symptoms1','symptoms2','symptoms3','symptoms4','symptoms5']
        input_df = pd.DataFrame([input_row], columns=cols)

        # Encode and predict
        encoded = encoder.transform(input_df)
        pred = model.predict(encoded)[0]
        proba = model.predict_proba(encoded)[0]

        # Display result
        label = 'Dangerous' if pred == 1 else 'Not Dangerous'
        st.metric(label="Prediction Result", value=label)
        st.write(f"**Probability** — Not Dangerous: {proba[0]:.2f}, Dangerous: {proba[1]:.2f}")

        # Remedial advice if dangerous
        if pred == 1:
            st.subheader("Recommended Actions Based on Symptoms")
            advice_map = {
    'vomiting': (
        "• Withhold food for 6–12 hours but continue to offer small amounts of water frequently.\n"
        "• If vomiting stops, reintroduce a bland diet (boiled rice and plain chicken) in small portions.\n"
        "• Keep the animal calm and restrict activity.\n"
        "• Clean up vomit promptly to prevent re-ingestion."
    ),
    'diarrhea': (
        "• Offer plenty of fresh water or unflavored electrolyte solution to prevent dehydration.\n"
        "• Withhold food for 12 hours (adults), then offer a bland diet (boiled rice and chicken).\n"
        "• Keep the animal’s hindquarters clean and dry.\n"
        "• Monitor for blood or mucus in the stool."
    ),
    'lethargy': (
        "• Provide a quiet, warm, and comfortable resting area.\n"
        "• Ensure easy access to water and food.\n"
        "• Limit exercise and monitor for worsening symptoms."
    ),
    'fever': (
        "• Keep the animal in a cool, shaded, and comfortable area.\n"
        "• Offer cool water frequently.\n"
        "• Avoid excessive handling or exercise."
    ),
    'coughing': (
        "• Keep the environment free from dust, smoke, and strong odors.\n"
        "• Use a humidifier or allow the animal to breathe in steam for 10 minutes.\n"
        "• Ensure access to fresh water."
    ),
    'dehydration': (
        "• Offer small amounts of water or unflavored electrolyte solution frequently.\n"
        "• Provide ice cubes if the animal won’t drink.\n"
        "• Keep the animal in a cool, shaded area."
    ),
    'loss of appetite': (
        "• Offer highly palatable, easily digestible foods (e.g., boiled chicken).\n"
        "• Warm food slightly to enhance aroma.\n"
        "• Hand-feed small amounts if necessary."
    ),
    'weight loss': (
        "• Feed small, frequent meals of high-calorie, nutrient-dense foods.\n"
        "• Weigh your animal regularly to track changes.\n"
        "• Check for parasites and maintain a regular deworming schedule."
    ),
    'nasal discharge': (
        "• Gently wipe the nose with a soft, damp cloth.\n"
        "• Use a humidifier or allow the animal to breathe in steam.\n"
        "• Keep the animal warm and away from drafts."
    ),
    'eye discharge': (
        "• Clean the area around the eyes with a soft, damp cotton ball.\n"
        "• Prevent rubbing or scratching the eyes.\n"
        "• Keep bedding and living areas clean."
    ),
    'limping': (
        "• Restrict activity and prevent jumping or running.\n"
        "• Apply a cold compress to the affected limb for 10–15 minutes.\n"
        "• Check paw pads for cuts, thorns, or foreign objects."
    ),
    'itching': (
        "• Check the skin and fur for fleas, ticks, or signs of irritation.\n"
        "• Bathe the animal with a gentle, hypoallergenic pet shampoo.\n"
        "• Keep nails trimmed."
    ),
    'hair loss': (
        "• Brush the coat regularly to remove loose hair and stimulate the skin.\n"
        "• Ensure a balanced diet with adequate protein, vitamins, and minerals.\n"
        "• Reduce stress and provide a consistent routine."
    ),
    'seizures': (
        "• Move objects away to prevent injury.\n"
        "• Do not restrain the animal or put anything in its mouth.\n"
        "• Keep the area quiet and dimly lit."
    ),
    'heatstroke': (
        "• Move the animal to a shaded, cool area immediately.\n"
        "• Offer small amounts of cool (not cold) water.\n"
        "• Wet the fur with cool water and use a fan to help with cooling."
    ),
    'hypothermia': (
        "• Move the animal to a warm, dry area away from drafts.\n"
        "• Wrap in warm blankets or towels; use warm water bottles wrapped in cloth.\n"
        "• Offer lukewarm fluids if the animal is alert."
    ),
    'swelling': (
        "• Apply a cold compress to the swollen area for 10–15 minutes.\n"
        "• Keep the animal calm and restrict movement.\n"
        "• Elevate the affected limb if possible."
    ),
    'bleeding': (
        "• Apply firm, direct pressure to the wound with a clean cloth or gauze.\n"
        "• Elevate the bleeding area above heart level if possible.\n"
        "• Once bleeding stops, cover with a clean bandage."
    ),
    'pain': (
        "• Create a quiet, comfortable resting area.\n"
        "• Avoid handling or touching the painful area.\n"
        "• Do not give human pain medications."
    ),
    'runny nose': (
        "• Wipe the nose gently with a soft, damp cloth.\n"
        "• Use a humidifier or allow the animal to inhale steam.\n"
        "• Keep the animal warm and away from drafts."
    ),
    'eye redness': (
        "• Prevent the animal from rubbing or scratching the eye.\n"
        "• Clean gently with a saline-moistened cotton pad.\n"
        "• Keep the animal in a dimly lit, calm environment."
    ),
    'abdominal pain': (
        "• Withhold food for several hours, but offer water if not vomiting.\n"
        "• Encourage rest in a quiet, comfortable area.\n"
        "• Avoid pressing or handling the abdomen."
    ),
    'constipation': (
        "• Provide access to fresh water at all times.\n"
        "• Encourage gentle exercise.\n"
        "• Offer canned pumpkin or a small amount of olive oil mixed with food."
    ),
    'hair loss': (
        "• Brush regularly to remove loose hair and stimulate the skin.\n"
        "• Ensure a balanced diet rich in omega-3 and omega-6 fatty acids.\n"
        "• Reduce stress and provide a consistent routine."
    ),
    'shivering': (
        "• Move the animal to a warm, dry area.\n"
        "• Wrap in a blanket and offer warm fluids if alert.\n"
        "• Avoid sudden temperature changes."
    ),
    'bad breath': (
        "• Brush the animal’s teeth regularly with pet-safe toothpaste.\n"
        "• Provide dental chews or toys to help clean teeth.\n"
        "• Ensure a balanced diet and avoid table scraps."
    ),
    'drooling': (
        "• Check for foreign objects in the mouth and remove if safe.\n"
        "• Offer cool water and keep the animal calm.\n"
        "• Avoid feeding until drooling resolves."
    ),
    'dandruff': (
        "• Brush the coat regularly to remove flakes.\n"
        "• Bathe with a moisturizing, pet-safe shampoo.\n"
        "• Ensure proper nutrition and hydration."
    ),
    'lumps, bumps': (
        "• Monitor for changes in size, shape, or color.\n"
        "• Avoid pressing or manipulating the lump.\n"
        "• Keep the area clean and dry."
    ),
    'staggering': (
        "• Restrict movement and keep the animal in a safe, quiet area.\n"
        "• Remove obstacles and hazards from the environment.\n"
        "• Offer water and observe for worsening symptoms."
    ),
    'blindness': (
        "• Keep the environment consistent and avoid moving furniture.\n"
        "• Use verbal cues and gentle touch to guide your pet.\n"
        "• Block off stairs and dangerous areas."
    ),
    'deafness': (
        "• Use visual signals or vibrations to communicate.\n"
        "• Approach your pet from the front to avoid startling.\n"
        "• Keep your pet on a leash outdoors."
    ),
    'straining to urinate': (
        "• Ensure access to clean, fresh water.\n"
        "• Monitor for successful urination and note any blood.\n"
        "• Keep the animal calm and restrict activity."
    ),
    'blood in urine': (
        "• Provide plenty of fresh water.\n"
        "• Monitor for frequency and volume of urination.\n"
        "• Keep the animal calm and restrict activity."
    ),
    'blood in stool': (
        "• Withhold food for 12 hours if diarrhea is present.\n"
        "• Offer water or electrolyte solution.\n"
        "• Clean the animal’s hindquarters frequently."
    ),
    'excessive thirst': (
        "• Ensure unlimited access to clean, fresh water.\n"
        "• Monitor water intake and urination frequency.\n"
        "• Avoid salty treats or foods."
    ),
    'excessive urination': (
        "• Ensure unlimited access to clean, fresh water.\n"
        "• Monitor for accidents and clean promptly.\n"
        "• Provide frequent bathroom breaks."
    ),
    'panting': (
        "• Move the animal to a cool, shaded area.\n"
        "• Offer cool water to drink.\n"
        "• Avoid exercise and excitement until panting subsides."
    ),
    'sneezing': (
        "• Keep the environment free of dust, smoke, and strong odors.\n"
        "• Use a humidifier if the air is dry.\n"
        "• Monitor for nasal discharge or blood."
    ),
    'scratching': (
        "• Check for fleas, ticks, or skin irritation.\n"
        "• Bathe with a gentle, hypoallergenic shampoo.\n"
        "• Keep nails trimmed to prevent self-injury."
    ),
    'shaking head': (
        "• Check ears for redness, swelling, or discharge.\n"
        "• Clean ears gently with a pet-safe ear cleaner.\n"
        "• Remove visible foreign objects if safe."
    ),
    'swollen abdomen': (
        "• Withhold food and water if vomiting or in distress.\n"
        "• Keep the animal calm and restrict movement.\n"
        "• Monitor for labored breathing or collapse."
    ),
    'difficulty breathing': (
        "• Move the animal to a well-ventilated, calm area.\n"
        "• Remove any tight collars or harnesses.\n"
        "• Keep the animal calm and avoid excitement."
    ),
    'pale gums': (
        "• Keep the animal calm and restrict activity.\n"
        "• Ensure access to water.\n"
        "• Monitor for signs of weakness or collapse."
    ),
    'wounds': (
        "• Clean the wound gently with saline or clean water.\n"
        "• Apply a clean bandage.\n"
        "• Prevent licking or scratching."
    ),
    'burns': (
        "• Flush the area with cool (not cold) water for several minutes.\n"
        "• Cover with a clean, nonstick bandage.\n"
        "• Keep the animal calm and restrict movement."
    ),
    'broken bones': (
        "• Minimize movement; use a board or blanket as a stretcher.\n"
        "• Immobilize the limb with padding and a rigid item (like a magazine).\n"
        "• Avoid manipulating the limb unnecessarily."
    ),
    'choking': (
        "• Open the mouth carefully and remove visible objects if safe.\n"
        "• Do not push objects deeper.\n"
        "• If the animal collapses, perform chest compressions or Heimlich maneuver as appropriate."
    ),
    'allergic reaction': (
        "• Remove the allergen if known (e.g., food, plant, chemical).\n"
        "• Bathe the animal if the allergen is on the skin.\n"
        "• Apply a cold compress to swollen areas."
    ),
    'heat stress': (
        "• Move to a cool, shaded area immediately.\n"
        "• Offer small amounts of cool water.\n"
        "• Wet fur with cool water and use a fan."
    ),
    'frostbite': (
        "• Move to a warm, dry area immediately.\n"
        "• Warm affected areas gently with lukewarm water.\n"
        "• Do not rub or massage frostbitten areas."
    ),
    'dizziness': (
        "• Keep the animal in a quiet, safe area.\n"
        "• Remove obstacles and hazards.\n"
        "• Offer water and monitor for worsening symptoms."
    ),
    'stiffness': (
        "• Encourage gentle movement and stretching.\n"
        "• Provide a warm, soft resting area.\n"
        "• Avoid strenuous activity."
    ),
    'muscle tremors': (
        "• Keep the animal calm and in a quiet area.\n"
        "• Offer water and monitor for worsening symptoms.\n"
        "• Avoid excitement or sudden movements."
    ),
    'collapse': (
        "• Keep the animal on its side in a quiet, safe area.\n"
        "• Ensure the airway is clear.\n"
        "• Monitor breathing and pulse."
    ),
    'paralysis': (
        "• Keep the animal still and comfortable.\n"
        "• Prevent movement of affected limbs.\n"
        "• Offer water if the animal can swallow."
    ),
    'aggression': (
        "• Isolate the animal from people and other pets.\n"
        "• Reduce environmental stressors.\n"
        "• Maintain a calm and consistent routine."
    ),
    'anxiety': (
        "• Provide a safe, quiet space.\n"
        "• Use calming aids (e.g., pheromone diffusers, soothing music).\n"
        "• Maintain a consistent routine."
    ),
    'excessive grooming': (
        "• Distract with toys or gentle play.\n"
        "• Keep the skin and coat clean.\n"
        "• Monitor for skin sores or bald patches."
    ),
    'loss of coordination': (
        "• Restrict movement and keep the animal in a safe area.\n"
        "• Remove hazards and obstacles.\n"
        "• Offer water and monitor for worsening symptoms."
    ),
    'difficulty walking': (
        "• Restrict activity and prevent jumping or running.\n"
        "• Provide a soft, non-slip surface for walking.\n"
        "• Support the animal when moving if needed."
    ),
    'loss of balance': (
        "• Keep the animal in a quiet, safe area.\n"
        "• Remove obstacles and hazards.\n"
        "• Avoid sudden movements or excitement."
    ),
    'trembling': (
        "• Move to a warm, quiet area.\n"
        "• Wrap in a blanket if cold.\n"
        "• Offer water and monitor for worsening symptoms."
    ),
    'shock': (
        "• Keep the animal as quiet and calm as possible.\n"
        "• Lay the animal on their side and cover with a blanket to conserve body heat (unless heatstroke is suspected).\n"
        "• Gently cover the eyes to reduce stress.\n"
        "• Elevate the hindquarters slightly if there are no obvious injuries.\n"
        "• Check airway, breathing, and pulse; clear any obstructions from the mouth if present.\n"
        "• Handle gently and minimize movement."
    ),
    'difficulty breathing': (
        "• Move the animal to a well-ventilated, quiet area with fresh air.\n"
        "• Remove any tight collars or harnesses.\n"
        "• Check the mouth for visible obstructions and carefully remove if safe.\n"
        "• Keep the animal calm and avoid excitement or exertion.\n"
        "• If not breathing and unconscious, perform rescue breathing: close the mouth and breathe into the nose every 3–5 seconds until breathing resumes or help arrives."
    ),
    'open mouth breathing/gasping': (
        "• Place the animal in a cool, quiet area with good airflow.\n"
        "• Avoid handling or stressing the animal further.\n"
        "• Check for obstructions in the mouth and clear if visible and safe.\n"
        "• Monitor closely for worsening signs."
    ),
    'loud respiration': (
        "• Keep the animal still and in a calm environment.\n"
        "• Remove potential irritants (smoke, dust, strong odors) from the area.\n"
        "• Observe for additional respiratory distress."
    ),
    'discharge from nose': (
        "• Gently wipe away discharge with a soft, damp cloth.\n"
        "• Use a humidifier or allow the animal to breathe in steam for a few minutes to loosen mucus.\n"
        "• Keep the animal warm and away from drafts."
    ),
    'bald patches/fluffed feathers': (
        "• Provide a stress-free, clean environment.\n"
        "• Ensure proper nutrition and regular grooming.\n"
        "• Avoid excessive handling and monitor for skin irritation."
    ),
    'wounds': (
        "• Approach the animal calmly and safely; use gloves if available.\n"
        "• Apply firm, direct pressure with a clean cloth or gauze to stop bleeding.\n"
        "• Rinse minor wounds with saline or clean water.\n"
        "• Cover with a sterile, nonstick bandage."
    ),
    'uncontrolled bleeding': (
        "• Apply firm, direct pressure to the wound with a clean towel or gauze for several minutes.\n"
        "• If possible, elevate the bleeding area above heart level.\n"
        "• Do not remove any deeply embedded objects; instead, pad around them and control bleeding as best as possible."
    ),
    'inability to walk': (
        "• Restrict all movement and keep the animal in a safe, padded area.\n"
        "• Gently support the animal on a board, blanket, or towel for transport if necessary.\n"
        "• Avoid manipulating the spine or limbs unnecessarily."
    ),
    'loss of consciousness': (
        "• Lay the animal on their side in a quiet, safe area.\n"
        "• Ensure the airway is clear; remove visible obstructions from the mouth.\n"
        "• Check breathing and pulse; if absent, begin CPR (chest compressions and rescue breaths as appropriate for species and size)."
    ),
    'major injuries/fractures': (
        "• Keep the animal as still as possible.\n"
        "• Do not attempt to splint or bandage the fracture; improper handling can worsen the injury.\n"
        "• Gently place the animal in a padded box or on a flat surface for transport.\n"
        "• Support the injured area with soft padding."
    ),
    'burns': (
        "• Flush the burn area with cool (not cold) running water for several minutes.\n"
        "• Do not apply creams, ointments, or ice.\n"
        "• Cover the burn with a clean, nonstick bandage or cloth.\n"
        "• Keep the animal warm and calm."
    ),
    'impalement': (
        "• Do not remove the object.\n"
        "• Stabilize the object with padding to prevent movement.\n"
        "• Control any bleeding around the object with gentle pressure.\n"
        "• Keep the animal calm and restrict movement."
    ),
    'seizures/fits': (
        "• Clear the area of objects to prevent injury.\n"
        "• Do not restrain the animal or put anything in its mouth.\n"
        "• Dim lights and keep the environment quiet.\n"
        "• Allow the animal to recover in a safe, comfortable space."
    ),
    'exposure (heatstroke)': (
        "• Move the animal to a cool, shaded area immediately.\n"
        "• Wet the fur with cool (not cold) water and use a fan to aid cooling.\n"
        "• Offer small amounts of cool water to drink."
    ),
    'exposure (cold/hypothermia)': (
        "• Move the animal to a warm, dry area.\n"
        "• Wrap in warm blankets or towels; use warm water bottles wrapped in cloth.\n"
        "• Avoid rapid reheating or direct heat sources."
    ),
    'poisoning': (
        "• Remove the animal from the source of poison.\n"
        "• Do not induce vomiting unless instructed by a professional.\n"
        "• Save packaging or a sample of the poison for identification.\n"
        "• Rinse skin or mouth with water if safe and appropriate."
    ),
    'ingestion of foreign objects': (
        "• Remove visible objects from the mouth if safe to do so.\n"
        "• Do not induce vomiting unless specifically advised.\n"
        "• Monitor for signs of choking, pain, or distress."
    ),
    'choking': (
        "• Open the mouth and sweep for visible obstructions with a finger or tweezers if safe.\n"
        "• Do not push objects deeper.\n"
        "• If the animal is not breathing, perform the Heimlich maneuver: for small animals, apply quick, gentle pressure just below the ribcage; for larger animals, use abdominal thrusts as described in pet first aid guides."
    ),
    'CPR/cardiac arrest': (
        "• Lay the animal on its side (on its back for wide-chested dogs).\n"
        "• For small animals, use one hand to compress the chest; for large animals, use both hands.\n"
        "• Apply 30 compressions at a rate of two per second, then give two rescue breaths by closing the mouth and blowing into the nose.\n"
        "• Alternate compressions and breaths until breathing or a heartbeat returns."
    ),
    'allergic reaction': (
        "• Remove the allergen if known (e.g., food, plant, chemical).\n"
        "• Bathe the animal if the allergen is on the skin.\n"
        "• Apply a cold compress to swollen areas.\n"
        "• Keep the animal calm and monitor for worsening symptoms."
    ),
    'pet allergy (in people)': (
        "• Create pet-free zones in the home.\n"
        "• Bathe pets weekly with pet-safe shampoo.\n"
        "• Remove carpets and use HEPA filters to reduce dander.\n"
        "• Rinse nasal passages with saline to reduce symptoms."
    ),
    'handling and restraint': (
        "• Always approach injured or scared animals slowly and calmly.\n"
        "• For small animals or birds, wrap gently in a towel or handling bag, covering the eyes to reduce stress.\n"
        "• For larger or aggressive animals, use two people if needed and appropriate restraint tools.\n"
        "• Ensure the animal can breathe and is not constricted."
    ),
    'bites/scratches': (
        "• Clean the area gently with saline or clean water.\n"
        "• Apply a clean, nonstick bandage.\n"
        "• Monitor for swelling, redness, or discharge."
    ),
    'convulsions': (
        "• Remove objects nearby to prevent injury.\n"
        "• Do not restrain the animal or put anything in its mouth.\n"
        "• Allow the animal to recover in a quiet, dimly lit area."
    ),
    'artificial respiration': (
        "• Check for breathing and pulse.\n"
        "• If not breathing, close the mouth and breathe into the nose every 3–5 seconds until breathing resumes.\n"
        "• Continue until the animal breathes on its own or help arrives."
    ),
    # Add more as needed from your original list using the same approach.
}

            for sym in selected_symptoms:
                advice = advice_map.get(sym.lower())
                if advice:
                    st.markdown(f"- **{sym.title()}**: {advice}")

