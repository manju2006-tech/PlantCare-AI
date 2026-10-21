"""
Plant disease knowledge base.

Maps a PlantVillage-style class name (e.g. "Tomato___Late_blight") to full,
human-readable information used to build the prediction result page and PDF
report: description, causes, treatment, prevention, fertilizer recommendation,
and pesticide recommendation.

This file covers a solid representative set of the most common PlantVillage
crops (Apple, Corn/Maize, Grape, Potato, Tomato, Pepper, Strawberry). Any
class name not found in DISEASE_INFO automatically falls back to
`_generic_entry()`, which builds a sensible, still-complete entry from the
class name itself - so the app never breaks even if you retrain with the
full 38-class PlantVillage dataset or your own custom classes. To extend
coverage, just add more keys to DISEASE_INFO following the same structure.
"""

DISEASE_INFO = {

    # ------------------------------------------------------------------ Apple
    "Apple___Apple_scab": {
        "plant_name": "Apple",
        "disease_name": "Apple Scab",
        "is_healthy": False,
        "description": (
            "Apple scab is a common fungal disease that produces dark, "
            "scabby lesions on leaves and fruit, reducing fruit quality "
            "and, in severe cases, causing early leaf drop."
        ),
        "causes": (
            "Caused by the fungus Venturia inaequalis. Spores overwinter in "
            "fallen leaves and spread to new growth during cool, wet spring "
            "weather. Prolonged leaf wetness strongly favors infection."
        ),
        "treatment": (
            "Remove and destroy infected leaves and fruit. Apply a "
            "fungicide labeled for apple scab (e.g. captan or myclobutanil) "
            "starting at bud break and repeating on a 7-10 day schedule "
            "during wet periods."
        ),
        "prevention": (
            "Rake and destroy fallen leaves each autumn, prune for good air "
            "circulation, choose scab-resistant apple varieties, and avoid "
            "overhead irrigation that keeps foliage wet."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (10-10-10) applied in early spring; avoid excess "
            "nitrogen, which promotes dense, disease-prone foliage."
        ),
        "pesticide_recommendation": (
            "Captan 50WP or Myclobutanil-based fungicide, applied per label "
            "instructions every 7-10 days during the wet spring season."
        ),
    },
    "Apple___Black_rot": {
        "plant_name": "Apple",
        "disease_name": "Black Rot",
        "is_healthy": False,
        "description": (
            "Black rot causes purple-bordered leaf spots, rotting fruit "
            "with concentric rings, and sunken cankers on branches."
        ),
        "causes": (
            "Caused by the fungus Botryosphaeria obtusa, which infects "
            "through wounds, dead wood, and old cankers, spreading rapidly "
            "in warm, humid conditions."
        ),
        "treatment": (
            "Prune out dead or cankered wood at least 6 inches below "
            "visible damage, remove mummified fruit, and apply a "
            "fungicide containing thiophanate-methyl or captan."
        ),
        "prevention": (
            "Sanitize the orchard by removing dead wood and mummified "
            "fruit, avoid tree stress, and maintain proper pruning for "
            "airflow."
        ),
        "fertilizer_recommendation": (
            "Balanced 10-10-10 NPK in spring plus calcium supplementation "
            "to strengthen fruit skin against infection."
        ),
        "pesticide_recommendation": (
            "Thiophanate-methyl or Captan fungicide, applied at pink bud "
            "stage and repeated through the growing season per label."
        ),
    },
    "Apple___Cedar_apple_rust": {
        "plant_name": "Apple",
        "disease_name": "Cedar Apple Rust",
        "is_healthy": False,
        "description": (
            "Produces bright orange-yellow spots on leaves and can deform "
            "fruit; requires a nearby juniper/cedar host to complete its "
            "life cycle."
        ),
        "causes": (
            "Caused by the fungus Gymnosporangium juniperi-virginianae, "
            "which alternates between apple/crabapple and juniper/cedar "
            "trees."
        ),
        "treatment": (
            "Apply a fungicide (myclobutanil or propiconazole) at "
            "pink-bud stage through petal fall. Remove nearby galls from "
            "juniper/cedar trees if possible."
        ),
        "prevention": (
            "Plant rust-resistant apple varieties and, where practical, "
            "avoid planting apples within a few hundred meters of "
            "juniper/cedar trees."
        ),
        "fertilizer_recommendation": (
            "Balanced 10-10-10 NPK in early spring to support vigorous, "
            "resilient growth."
        ),
        "pesticide_recommendation": (
            "Myclobutanil or Propiconazole fungicide sprays from pink bud "
            "through petal fall, per label directions."
        ),
    },
    "Apple___healthy": {
        "plant_name": "Apple",
        "disease_name": "Healthy",
        "is_healthy": True,
        "description": "This apple leaf shows no visible signs of disease and appears healthy.",
        "causes": "Not applicable - no disease detected.",
        "treatment": "No treatment needed. Continue routine care.",
        "prevention": (
            "Maintain regular watering, balanced fertilization, good "
            "airflow through pruning, and routine monitoring for early "
            "signs of pests or disease."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (10-10-10) in early spring and again in early "
            "summer to sustain healthy growth."
        ),
        "pesticide_recommendation": "None required for a healthy plant; monitor periodically.",
    },

    # -------------------------------------------------------------- Corn/Maize
    "Corn_(maize)___Common_rust_": {
        "plant_name": "Corn (Maize)",
        "disease_name": "Common Rust",
        "is_healthy": False,
        "description": (
            "Small, reddish-brown, elongated pustules appear on both leaf "
            "surfaces, which can reduce photosynthesis and yield in heavy "
            "infections."
        ),
        "causes": (
            "Caused by the fungus Puccinia sorghi. Spores spread by wind "
            "and thrive in cool, moist conditions (temperatures 60-77°F "
            "with high humidity)."
        ),
        "treatment": (
            "Apply a foliar fungicide containing azoxystrobin or "
            "propiconazole if rust appears before tasseling and coverage "
            "is severe."
        ),
        "prevention": (
            "Plant rust-resistant hybrids, rotate crops, and avoid "
            "excessive nitrogen which promotes dense canopies retaining "
            "moisture."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK with moderate nitrogen (e.g. 20-10-10) split "
            "across planting and knee-high growth stage."
        ),
        "pesticide_recommendation": (
            "Azoxystrobin or Propiconazole fungicide, applied at first "
            "sign of pustules and repeated per label interval."
        ),
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant_name": "Corn (Maize)",
        "disease_name": "Northern Leaf Blight",
        "is_healthy": False,
        "description": (
            "Long, elliptical, gray-green to tan lesions form on leaves, "
            "which can merge and cause significant leaf blighting and "
            "yield loss."
        ),
        "causes": (
            "Caused by the fungus Exserohilum turcicum, favored by "
            "moderate temperatures (64-80°F), heavy dew, and extended leaf "
            "wetness."
        ),
        "treatment": (
            "Apply a fungicide containing strobilurin or triazole active "
            "ingredients at first sign of lesions, especially before "
            "tasseling."
        ),
        "prevention": (
            "Rotate with non-host crops, till under crop residue after "
            "harvest, and plant resistant hybrids where available."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (20-10-10) with adequate potassium to improve "
            "stalk and leaf resilience."
        ),
        "pesticide_recommendation": (
            "Strobilurin or Triazole-based fungicide, applied at early "
            "lesion onset, repeated per label directions."
        ),
    },
    "Corn_(maize)___healthy": {
        "plant_name": "Corn (Maize)",
        "disease_name": "Healthy",
        "is_healthy": True,
        "description": "This corn leaf shows no visible signs of disease and appears healthy.",
        "causes": "Not applicable - no disease detected.",
        "treatment": "No treatment needed. Continue routine care.",
        "prevention": (
            "Maintain crop rotation, balanced fertilization, and regular "
            "field scouting for early pest or disease detection."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (20-10-10) split between planting and knee-high "
            "growth stage."
        ),
        "pesticide_recommendation": "None required for a healthy plant; monitor periodically.",
    },

    # ------------------------------------------------------------------ Grape
    "Grape___Black_rot": {
        "plant_name": "Grape",
        "disease_name": "Black Rot",
        "is_healthy": False,
        "description": (
            "Causes small tan leaf spots with dark borders and turns "
            "berries into hard, black, shriveled 'mummies'."
        ),
        "causes": (
            "Caused by the fungus Guignardia bidwellii, which overwinters "
            "in mummified berries and infected canes, spreading in warm, "
            "wet weather."
        ),
        "treatment": (
            "Remove and destroy mummified berries and infected canes; "
            "apply a fungicide (myclobutanil or mancozeb) starting at "
            "bud break."
        ),
        "prevention": (
            "Prune for open canopy airflow, remove mummies and infected "
            "wood during dormant pruning, and avoid overhead irrigation."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (10-10-10) in early spring; avoid excess "
            "nitrogen which increases canopy density."
        ),
        "pesticide_recommendation": (
            "Myclobutanil or Mancozeb fungicide from bud break through "
            "veraison, per label schedule."
        ),
    },
    "Grape___healthy": {
        "plant_name": "Grape",
        "disease_name": "Healthy",
        "is_healthy": True,
        "description": "This grape leaf shows no visible signs of disease and appears healthy.",
        "causes": "Not applicable - no disease detected.",
        "treatment": "No treatment needed. Continue routine care.",
        "prevention": (
            "Maintain canopy management for airflow, balanced "
            "fertilization, and routine vineyard scouting."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (10-10-10) in early spring with moderate "
            "potassium during fruit development."
        ),
        "pesticide_recommendation": "None required for a healthy plant; monitor periodically.",
    },

    # ---------------------------------------------------------------- Potato
    "Potato___Early_blight": {
        "plant_name": "Potato",
        "disease_name": "Early Blight",
        "is_healthy": False,
        "description": (
            "Produces dark brown spots with concentric 'target' rings on "
            "older leaves first, which can spread and reduce tuber yield."
        ),
        "causes": (
            "Caused by the fungus Alternaria solani, favored by warm "
            "temperatures, alternating wet and dry conditions, and plant "
            "stress or nutrient deficiency."
        ),
        "treatment": (
            "Remove severely infected leaves and apply a fungicide "
            "containing chlorothalonil or mancozeb at first symptom onset."
        ),
        "prevention": (
            "Rotate crops on a 2-3 year cycle, avoid overhead watering, "
            "ensure balanced fertility, and space plants for airflow."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (14-14-14) with adequate potassium to reduce "
            "plant stress and improve resistance."
        ),
        "pesticide_recommendation": (
            "Chlorothalonil or Mancozeb fungicide, applied every 7-10 days "
            "once symptoms appear."
        ),
    },
    "Potato___Late_blight": {
        "plant_name": "Potato",
        "disease_name": "Late Blight",
        "is_healthy": False,
        "description": (
            "A fast-moving, destructive disease causing dark, water-soaked "
            "lesions on leaves and stems that can destroy a crop within "
            "days under favorable conditions."
        ),
        "causes": (
            "Caused by the oomycete Phytophthora infestans (the same "
            "pathogen behind the Irish potato famine), spreading rapidly in "
            "cool, wet, humid weather."
        ),
        "treatment": (
            "Remove and destroy infected plants immediately; apply a "
            "systemic fungicide containing chlorothalonil, mancozeb, or "
            "metalaxyl without delay."
        ),
        "prevention": (
            "Plant certified disease-free seed potatoes, avoid overhead "
            "irrigation, ensure good drainage and airflow, and destroy "
            "volunteer potato plants."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (14-14-14) avoiding excess nitrogen, which "
            "increases susceptibility to late blight."
        ),
        "pesticide_recommendation": (
            "Metalaxyl + Mancozeb combination fungicide, applied "
            "preventively every 5-7 days during humid weather."
        ),
    },
    "Potato___healthy": {
        "plant_name": "Potato",
        "disease_name": "Healthy",
        "is_healthy": True,
        "description": "This potato leaf shows no visible signs of disease and appears healthy.",
        "causes": "Not applicable - no disease detected.",
        "treatment": "No treatment needed. Continue routine care.",
        "prevention": (
            "Use certified seed potatoes, rotate crops, and monitor "
            "regularly for early signs of blight or pests."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (14-14-14) split between planting and hilling "
            "stage."
        ),
        "pesticide_recommendation": "None required for a healthy plant; monitor periodically.",
    },

    # --------------------------------------------------------------- Tomato
    "Tomato___Bacterial_spot": {
        "plant_name": "Tomato",
        "disease_name": "Bacterial Spot",
        "is_healthy": False,
        "description": (
            "Small, dark, water-soaked spots appear on leaves, stems, and "
            "fruit, which can turn scabby and reduce marketable yield."
        ),
        "causes": (
            "Caused by Xanthomonas bacteria species, spread by splashing "
            "water, contaminated tools, and warm, humid, wet weather."
        ),
        "treatment": (
            "Remove infected plant debris, apply copper-based bactericide "
            "sprays, and avoid working in fields when foliage is wet."
        ),
        "prevention": (
            "Use disease-free seed/transplants, rotate crops, avoid "
            "overhead irrigation, and sanitize tools between plants."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (10-10-10) avoiding excess nitrogen, which "
            "produces soft, disease-susceptible growth."
        ),
        "pesticide_recommendation": (
            "Copper hydroxide or copper oxychloride bactericide, applied "
            "every 7 days during wet weather."
        ),
    },
    "Tomato___Early_blight": {
        "plant_name": "Tomato",
        "disease_name": "Early Blight",
        "is_healthy": False,
        "description": (
            "Dark spots with concentric rings form on older, lower leaves "
            "first, often surrounded by yellowing, and can progress "
            "upward through the plant."
        ),
        "causes": (
            "Caused by the fungus Alternaria solani, favored by warm "
            "temperatures, humidity, and nutrient-stressed plants."
        ),
        "treatment": (
            "Remove and destroy affected lower leaves, mulch to prevent "
            "soil splash, and apply a chlorothalonil or mancozeb "
            "fungicide."
        ),
        "prevention": (
            "Rotate crops, stake plants for airflow, avoid overhead "
            "watering, and maintain balanced soil fertility."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (10-10-10) with supplemental calcium to "
            "strengthen cell walls against infection."
        ),
        "pesticide_recommendation": (
            "Chlorothalonil or Mancozeb fungicide, applied every 7-10 "
            "days once symptoms appear."
        ),
    },
    "Tomato___Late_blight": {
        "plant_name": "Tomato",
        "disease_name": "Late Blight",
        "is_healthy": False,
        "description": (
            "Large, irregular, water-soaked, greasy-looking lesions "
            "spread rapidly across leaves, stems, and fruit, often with "
            "white fungal growth on the underside in humid conditions."
        ),
        "causes": (
            "Caused by the oomycete Phytophthora infestans, which spreads "
            "explosively in cool, wet, humid weather."
        ),
        "treatment": (
            "Remove and destroy infected plants immediately to stop "
            "spread; apply a systemic fungicide containing chlorothalonil "
            "or metalaxyl."
        ),
        "prevention": (
            "Avoid overhead irrigation, space plants for airflow, use "
            "resistant varieties, and destroy volunteer tomato/potato "
            "plants nearby."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (10-10-10) avoiding excess nitrogen; add "
            "potassium to improve disease resilience."
        ),
        "pesticide_recommendation": (
            "Chlorothalonil or Metalaxyl + Mancozeb fungicide, applied "
            "preventively every 5-7 days in humid weather."
        ),
    },
    "Tomato___Leaf_Mold": {
        "plant_name": "Tomato",
        "disease_name": "Leaf Mold",
        "is_healthy": False,
        "description": (
            "Pale green-yellow spots appear on upper leaf surfaces with "
            "olive-green to grayish-purple velvety mold underneath, "
            "common in humid greenhouse conditions."
        ),
        "causes": (
            "Caused by the fungus Passalora fulva (formerly Fulvia "
            "fulva), thriving in high humidity (above 85%) and poor "
            "ventilation."
        ),
        "treatment": (
            "Improve ventilation and reduce humidity, remove infected "
            "leaves, and apply a chlorothalonil or copper-based "
            "fungicide."
        ),
        "prevention": (
            "Increase plant spacing, use fans/ventilation in greenhouses, "
            "avoid overhead watering, and choose resistant varieties."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (10-10-10) with attention to airflow rather "
            "than heavy feeding."
        ),
        "pesticide_recommendation": (
            "Chlorothalonil-based fungicide, applied every 7-10 days "
            "while humidity remains high."
        ),
    },
    "Tomato___Septoria_leaf_spot": {
        "plant_name": "Tomato",
        "disease_name": "Septoria Leaf Spot",
        "is_healthy": False,
        "description": (
            "Numerous small, circular spots with dark borders and tan "
            "centers develop on lower leaves first, causing progressive "
            "defoliation."
        ),
        "causes": (
            "Caused by the fungus Septoria lycopersici, spreading via "
            "water splash in warm, wet, humid conditions."
        ),
        "treatment": (
            "Remove infected lower leaves, mulch soil to prevent splash, "
            "and apply chlorothalonil or copper-based fungicide."
        ),
        "prevention": (
            "Rotate crops, stake and prune for airflow, avoid overhead "
            "irrigation, and clear plant debris after harvest."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (10-10-10) to maintain plant vigor without "
            "excess foliage density."
        ),
        "pesticide_recommendation": (
            "Chlorothalonil fungicide, applied every 7-10 days once "
            "spots appear."
        ),
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant_name": "Tomato",
        "disease_name": "Tomato Yellow Leaf Curl Virus",
        "is_healthy": False,
        "description": (
            "Leaves curl upward and turn yellow, plants become stunted "
            "with reduced fruit set - one of the most damaging viral "
            "diseases of tomato."
        ),
        "causes": (
            "Caused by a Begomovirus transmitted by the whitefly "
            "(Bemisia tabaci); there is no cure once a plant is infected."
        ),
        "treatment": (
            "There is no chemical cure. Remove and destroy infected "
            "plants promptly to reduce virus reservoirs, and control the "
            "whitefly vector."
        ),
        "prevention": (
            "Use virus-resistant tomato varieties, reflective mulches to "
            "deter whiteflies, insect netting, and control whitefly "
            "populations early in the season."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (10-10-10) to support overall plant health; "
            "fertilizer alone will not control the virus."
        ),
        "pesticide_recommendation": (
            "Insecticidal soap, neem oil, or imidacloprid to control the "
            "whitefly vector - not the virus itself."
        ),
    },
    "Tomato___healthy": {
        "plant_name": "Tomato",
        "disease_name": "Healthy",
        "is_healthy": True,
        "description": "This tomato leaf shows no visible signs of disease and appears healthy.",
        "causes": "Not applicable - no disease detected.",
        "treatment": "No treatment needed. Continue routine care.",
        "prevention": (
            "Stake or cage plants for airflow, rotate crops annually, "
            "water at the base, and scout regularly for pests and early "
            "disease signs."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (10-10-10) at planting, switching to a "
            "lower-nitrogen, higher-potassium feed once fruiting begins."
        ),
        "pesticide_recommendation": "None required for a healthy plant; monitor periodically.",
    },

    # ------------------------------------------------------------ Bell Pepper
    "Pepper__bell___Bacterial_spot": {
        "plant_name": "Bell Pepper",
        "disease_name": "Bacterial Spot",
        "is_healthy": False,
        "description": (
            "Small, water-soaked spots appear on leaves and fruit that "
            "turn brown and scabby, and can cause significant leaf drop."
        ),
        "causes": (
            "Caused by Xanthomonas campestris pv. vesicatoria, spread by "
            "splashing water, wind-driven rain, and contaminated tools or "
            "seed."
        ),
        "treatment": (
            "Remove infected foliage, avoid overhead watering, and apply "
            "a copper-based bactericide, ideally combined with mancozeb "
            "for resistance management."
        ),
        "prevention": (
            "Use certified disease-free seed, rotate crops, sanitize "
            "tools, and avoid working with plants while foliage is wet."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (10-10-10) avoiding excess nitrogen which "
            "produces succulent, infection-prone tissue."
        ),
        "pesticide_recommendation": (
            "Copper hydroxide bactericide, applied every 7 days during "
            "wet or humid weather."
        ),
    },
    "Pepper__bell___healthy": {
        "plant_name": "Bell Pepper",
        "disease_name": "Healthy",
        "is_healthy": True,
        "description": "This bell pepper leaf shows no visible signs of disease and appears healthy.",
        "causes": "Not applicable - no disease detected.",
        "treatment": "No treatment needed. Continue routine care.",
        "prevention": (
            "Rotate crops, water at the base, stake plants for airflow, "
            "and scout regularly for early pest or disease signs."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (10-10-10) at planting with a potassium boost "
            "during fruit set."
        ),
        "pesticide_recommendation": "None required for a healthy plant; monitor periodically.",
    },

    # ---------------------------------------------------------------- Strawberry
    "Strawberry___Leaf_scorch": {
        "plant_name": "Strawberry",
        "disease_name": "Leaf Scorch",
        "is_healthy": False,
        "description": (
            "Small purple spots merge into larger scorched-looking "
            "blotches on leaves, which can reduce plant vigor and fruit "
            "yield."
        ),
        "causes": (
            "Caused by the fungus Diplocarpon earlianum, favored by warm, "
            "wet weather and dense, poorly ventilated plantings."
        ),
        "treatment": (
            "Remove and destroy infected leaves after harvest, and apply "
            "a fungicide containing captan or myclobutanil."
        ),
        "prevention": (
            "Space plants for airflow, avoid overhead irrigation, remove "
            "old infected foliage after harvest, and choose resistant "
            "varieties."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (10-10-10) applied after renovation/harvest to "
            "support regrowth."
        ),
        "pesticide_recommendation": (
            "Captan or Myclobutanil fungicide, applied every 7-14 days "
            "during active disease pressure."
        ),
    },
    "Strawberry___healthy": {
        "plant_name": "Strawberry",
        "disease_name": "Healthy",
        "is_healthy": True,
        "description": "This strawberry leaf shows no visible signs of disease and appears healthy.",
        "causes": "Not applicable - no disease detected.",
        "treatment": "No treatment needed. Continue routine care.",
        "prevention": (
            "Space plants for airflow, remove old foliage after harvest, "
            "and monitor regularly for early signs of leaf spot or mold."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (10-10-10) after renovation, with light feeding "
            "through the growing season."
        ),
        "pesticide_recommendation": "None required for a healthy plant; monitor periodically.",
    },
}


def _humanize(text: str) -> str:
    """Turn 'Northern_Leaf_Blight' or 'bell' into 'Northern Leaf Blight'."""
    return text.replace('_', ' ').replace('-', ' ').strip().title()


def _generic_entry(class_name: str) -> dict:
    """
    Safe fallback for any class name not explicitly listed in DISEASE_INFO.
    Parses the PlantVillage 'Plant___Disease' naming convention to still
    produce a complete, sensible result instead of crashing.
    """
    if '___' in class_name:
        plant_part, disease_part = class_name.split('___', 1)
    else:
        plant_part, disease_part = class_name, 'Unknown'

    plant_name = _humanize(plant_part)
    disease_name = _humanize(disease_part)
    is_healthy = 'healthy' in disease_part.lower()

    if is_healthy:
        return {
            "plant_name": plant_name,
            "disease_name": "Healthy",
            "is_healthy": True,
            "description": f"This {plant_name.lower()} leaf appears healthy with no visible signs of disease.",
            "causes": "Not applicable - no disease detected.",
            "treatment": "No treatment needed. Continue routine care.",
            "prevention": "Maintain balanced watering, fertilization, and regular monitoring.",
            "fertilizer_recommendation": "Balanced NPK (10-10-10) following standard crop schedule.",
            "pesticide_recommendation": "None required for a healthy plant; monitor periodically.",
        }

    return {
        "plant_name": plant_name,
        "disease_name": disease_name,
        "is_healthy": False,
        "description": (
            f"The model detected symptoms consistent with {disease_name} on "
            f"this {plant_name.lower()} leaf. Detailed reference information "
            f"for this specific class has not yet been added to the "
            f"knowledge base."
        ),
        "causes": (
            "Specific pathogen data is not yet available for this class. "
            "Common causes of leaf disease include fungal or bacterial "
            "pathogens spread by moisture, wind, or contaminated tools."
        ),
        "treatment": (
            "Isolate the affected plant, remove visibly damaged leaves, "
            "and consult a local agricultural extension office or "
            "certified agronomist for a targeted treatment plan."
        ),
        "prevention": (
            "General best practices: rotate crops, avoid overhead "
            "watering, ensure good airflow, and sanitize tools between "
            "plants."
        ),
        "fertilizer_recommendation": (
            "Balanced NPK (10-10-10) unless a soil test indicates a "
            "specific deficiency."
        ),
        "pesticide_recommendation": (
            "Consult a local extension service to identify the exact "
            "pathogen before selecting a targeted fungicide/bactericide."
        ),
    }


def get_disease_info(class_name: str) -> dict:
    """
    Main lookup function used by the prediction pipeline.
    Always returns a complete dict - never raises or returns None.
    """
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]
    return _generic_entry(class_name)


def list_all_classes():
    """Sorted list of class names with detailed knowledge-base entries."""
    return sorted(DISEASE_INFO.keys())