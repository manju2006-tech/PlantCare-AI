"""
Plant disease knowledge base.

Maps a model class name (e.g. "Tomato___Late_blight") to full, plain-English
information used to build the prediction result page and the PDF report:
description, causes, treatment, prevention, fertilizer recommendation and
pesticide recommendation.

All 38 classes produced by the trained model (see
trained_model/class_indices.json) are covered here with their EXACT class
names - including the ones that contain commas, brackets or spaces, such as
"Pepper,_bell___Bacterial_spot" and
"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot".

Each entry also carries plant_name / disease_name / is_healthy, because
routes/prediction_routes.py reads those three keys as well when it saves a
prediction. Removing them would break the prediction flow.

Language is deliberately simple so that someone with no farming or plant
science background can follow the advice. _generic_entry() remains as a
safety net for any future class name that is not listed here.
"""

DISEASE_INFO = {

    # ---------------------------------------------------------------- Apple
    "Apple___Apple_scab": {
        "plant_name": "Apple",
        "disease_name": "Apple Scab",
        "is_healthy": False,
        "description": (
            "Apple scab is a very common leaf and fruit disease caused by a "
            "fungus. It makes dark, rough, velvety spots on the leaves and "
            "scabby patches on the apples. Badly affected leaves turn yellow "
            "and fall off early, and the fruit looks damaged even though it "
            "is usually still safe to eat."
        ),
        "causes": (
            "The fungus survives the winter inside old apple leaves lying on "
            "the ground. In spring, rain splashes its spores up onto the new "
            "leaves. The disease spreads fast when the weather is cool and "
            "wet and the leaves stay damp for a long time."
        ),
        "treatment": (
            "Pick off and destroy the spotted leaves and fruit you can reach "
            "instead of leaving them under the tree. Spray the tree with a "
            "fungicide made for apple scab, starting when the buds begin to "
            "open and repeating every 7 to 10 days while the weather stays "
            "wet."
        ),
        "prevention": (
            "Rake up and remove all fallen leaves in autumn, because that is "
            "where the fungus hides over winter. Prune the tree so air can "
            "move freely through the branches and the leaves dry quickly. "
            "Water at the base of the tree rather than over the leaves, and "
            "choose scab-resistant apple varieties when planting new trees."
        ),
        "fertilizer_recommendation": (
            "Give a balanced fertiliser such as 10-10-10 once in early "
            "spring. Do not overfeed with nitrogen-rich fertiliser, because "
            "it pushes out a lot of soft new leaves that get infected more "
            "easily."
        ),
        "pesticide_recommendation": (
            "Use a fungicide containing captan or myclobutanil, following "
            "the instructions on the label. Spray every 7 to 10 days through "
            "the wet spring period, and always wear gloves and a mask while "
            "spraying."
        ),
    },
    "Apple___Black_rot": {
        "plant_name": "Apple",
        "disease_name": "Black Rot",
        "is_healthy": False,
        "description": (
            "Black rot is a fungal disease that attacks the leaves, the fruit "
            "and the branches. Leaves get small brown spots with purple "
            "edges, the apples develop dark rotten rings, and the branches "
            "can form sunken, cracked patches of dead bark."
        ),
        "causes": (
            "The fungus lives in dead wood, old cracked bark and dried-up "
            "fruit still hanging on the tree. It gets inside through cuts, "
            "broken branches and insect damage, and spreads quickly in warm, "
            "damp weather."
        ),
        "treatment": (
            "Cut out any dead or cracked branches, removing at least 15 cm "
            "(6 inches) below the damaged part, and burn or bin the "
            "cuttings. Pull off the dried, shrivelled apples still hanging "
            "on the tree. Then spray a suitable fungicide to protect the "
            "healthy parts."
        ),
        "prevention": (
            "Keep the tree clean: remove dead wood and old dried fruit every "
            "year. Avoid injuring the trunk and branches with tools. Keep the "
            "tree healthy with proper watering, because a stressed tree gets "
            "infected much more easily."
        ),
        "fertilizer_recommendation": (
            "Apply a balanced 10-10-10 fertiliser in spring. Adding calcium "
            "(for example gypsum or a calcium spray) helps build stronger "
            "fruit skin, which makes it harder for the disease to get in."
        ),
        "pesticide_recommendation": (
            "Use a fungicide containing thiophanate-methyl or captan. Start "
            "when the flower buds show pink colour and repeat through the "
            "growing season as the label directs."
        ),
    },
    "Apple___Cedar_apple_rust": {
        "plant_name": "Apple",
        "disease_name": "Cedar Apple Rust",
        "is_healthy": False,
        "description": (
            "This fungal disease makes bright orange or yellow spots on apple "
            "leaves, often with tiny raised bumps underneath. Heavily spotted "
            "leaves drop early, and the fruit can become small or misshapen."
        ),
        "causes": (
            "This fungus needs two different trees to survive. It spends part "
            "of its life on nearby cedar or juniper trees, where it forms "
            "brown balls called galls. In spring these galls release spores "
            "that blow onto apple trees during wet weather."
        ),
        "treatment": (
            "Spray the apple tree with a suitable fungicide from the time the "
            "flower buds turn pink until the petals fall. If there are cedar "
            "or juniper trees nearby, cut off and destroy the brown galls on "
            "them in late winter to reduce the spores."
        ),
        "prevention": (
            "Plant apple varieties that are described as rust-resistant. "
            "Where possible, do not plant apple trees close to cedar or "
            "juniper trees, since the disease needs both to complete its "
            "cycle."
        ),
        "fertilizer_recommendation": (
            "A balanced 10-10-10 fertiliser in early spring is enough. Strong, "
            "steady growth helps the tree cope with the leaf damage."
        ),
        "pesticide_recommendation": (
            "Use a fungicide containing myclobutanil or propiconazole. Spray "
            "from the pink bud stage until the flower petals have fallen, "
            "following the label timing."
        ),
    },
    "Apple___healthy": {
        "plant_name": "Apple",
        "disease_name": "Healthy",
        "is_healthy": True,
        "description": (
            "Good news - this apple leaf looks healthy. There are no spots, "
            "no discoloured patches and no signs of disease or insect damage."
        ),
        "causes": (
            "No disease was found, so there is nothing to explain here. The "
            "leaf simply looks the way a healthy apple leaf should."
        ),
        "treatment": (
            "No treatment is needed. Just carry on with your normal care and "
            "keep an eye on the tree."
        ),
        "prevention": (
            "Water the tree regularly but do not let the soil stay soggy. "
            "Prune it each year so air and sunlight can reach all the "
            "branches. Clear away fallen leaves in autumn, and check the "
            "leaves every couple of weeks so you catch any problem early."
        ),
        "fertilizer_recommendation": (
            "Feed with a balanced 10-10-10 fertiliser once in early spring "
            "and again in early summer. A soil test every few years tells you "
            "if anything specific is missing."
        ),
        "pesticide_recommendation": (
            "No spray is needed for a healthy tree. Only consider spraying if "
            "you actually see a pest or disease problem appear later."
        ),
    },

    # ------------------------------------------------------------ Blueberry
    "Blueberry___healthy": {
        "plant_name": "Blueberry",
        "disease_name": "Healthy",
        "is_healthy": True,
        "description": (
            "This blueberry leaf looks healthy. The colour is even and there "
            "are no spots, holes or dried patches that would suggest disease."
        ),
        "causes": (
            "No disease was found, so there is nothing to explain here."
        ),
        "treatment": (
            "No treatment is needed. Keep doing what you are already doing."
        ),
        "prevention": (
            "Blueberries need acidic soil, so keep the soil pH between about "
            "4.5 and 5.5 - this is the single most important thing for them. "
            "Use a mulch of pine bark or wood chips to hold moisture, water "
            "regularly, and prune out old or crowded stems each winter so air "
            "can move through the bush."
        ),
        "fertilizer_recommendation": (
            "Use a fertiliser made for acid-loving plants (the label may say "
            "for azaleas, rhododendrons or blueberries), applied in early "
            "spring. Avoid ordinary garden fertilisers containing lime, as "
            "they raise the soil pH and make blueberries suffer."
        ),
        "pesticide_recommendation": (
            "No spray is needed for a healthy plant. Netting is more useful "
            "than any chemical here, since birds are usually the main problem "
            "with blueberries."
        ),
    },

    # --------------------------------------------------------------- Cherry
    "Cherry_(including_sour)___Powdery_mildew": {
        "plant_name": "Cherry",
        "disease_name": "Powdery Mildew",
        "is_healthy": False,
        "description": (
            "Powdery mildew looks exactly like its name - a white or grey "
            "dusty coating on the leaves, as if someone sprinkled flour on "
            "them. Affected leaves may curl, twist or stay small, and heavy "
            "infections can also mark the fruit."
        ),
        "causes": (
            "A fungus causes this, and unlike most leaf diseases it does not "
            "need rain. It does best in warm days with cool, humid nights, "
            "especially where branches are crowded and air cannot move. Young "
            "soft leaves are the most likely to be attacked."
        ),
        "treatment": (
            "Cut off and destroy the worst affected shoots and leaves. Spray "
            "with a fungicide suitable for powdery mildew. Sulphur-based "
            "sprays and potassium bicarbonate sprays both work well if you "
            "prefer a gentler option."
        ),
        "prevention": (
            "Prune the tree to open up the centre so air flows through and "
            "sunlight reaches the inner leaves - crowding is the main thing "
            "that lets this disease take hold. Do not give too much nitrogen "
            "fertiliser, because soft new growth is the fungus's favourite "
            "target."
        ),
        "fertilizer_recommendation": (
            "Use a balanced 10-10-10 fertiliser in early spring, and keep the "
            "amount moderate. Too much nitrogen creates lots of soft leaves "
            "that get mildew easily."
        ),
        "pesticide_recommendation": (
            "Use a sulphur-based fungicide, potassium bicarbonate, or a "
            "product containing myclobutanil. Spray at the first white "
            "patches and repeat every 7 to 14 days as the label says. Do not "
            "use sulphur in very hot weather, as it can burn the leaves."
        ),
    },
    "Cherry_(including_sour)___healthy": {
        "plant_name": "Cherry",
        "disease_name": "Healthy",
        "is_healthy": True,
        "description": (
            "This cherry leaf looks healthy. There is no white dusty coating, "
            "no spots and no yellowing - the leaf appears normal."
        ),
        "causes": (
            "No disease was found, so there is nothing to explain here."
        ),
        "treatment": (
            "No treatment is needed. Continue your usual care routine."
        ),
        "prevention": (
            "Prune the tree while it is dormant in winter so the middle stays "
            "open and airy. Clear fallen leaves and old fruit from under the "
            "tree. Water at the base rather than over the leaves, and check "
            "the tree regularly for the first white patches of mildew."
        ),
        "fertilizer_recommendation": (
            "Apply a balanced 10-10-10 fertiliser once in early spring. "
            "Cherries do not need heavy feeding, so avoid giving extra "
            "nitrogen through the season."
        ),
        "pesticide_recommendation": (
            "No spray is needed while the tree is healthy. Just keep checking "
            "the leaves so you can act early if mildew appears."
        ),
    },

    # ----------------------------------------------------------- Corn/Maize
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant_name": "Corn (Maize)",
        "disease_name": "Gray Leaf Spot",
        "is_healthy": False,
        "description": (
            "Gray leaf spot makes long, narrow, grey or tan rectangular marks "
            "on corn leaves. The marks have very straight edges because they "
            "are boxed in by the leaf veins. When there are many of them they "
            "join together and the whole leaf dries out, which cuts down the "
            "size of the cobs."
        ),
        "causes": (
            "A fungus causes this, and it survives on old corn stalks and "
            "leaves left in the field after harvest. It starts on the lower "
            "leaves and moves upward. Warm, humid weather with heavy dew or "
            "frequent rain makes it spread quickly."
        ),
        "treatment": (
            "If the spots appear on the leaves near the cob before the plant "
            "flowers, spray a suitable fungicide - that is when it is worth "
            "doing. After harvest, plough the old stalks into the soil or "
            "remove them so the fungus has nowhere to survive."
        ),
        "prevention": (
            "Do not plant corn in the same field year after year - rotate "
            "with a different crop such as beans or wheat for a season. Bury "
            "or remove old corn debris after harvest, choose seed described "
            "as resistant to gray leaf spot, and avoid planting rows too "
            "close together."
        ),
        "fertilizer_recommendation": (
            "Use a balanced fertiliser such as 20-10-10, split between "
            "planting time and when the plants are knee-high. Make sure there "
            "is enough potassium, as it helps the plant stay strong and cope "
            "with leaf damage."
        ),
        "pesticide_recommendation": (
            "Use a fungicide containing azoxystrobin, pyraclostrobin or "
            "propiconazole. The best time is just before or at flowering, "
            "when the leaves around the cob need protecting."
        ),
    },
    "Corn_(maize)___Common_rust_": {
        "plant_name": "Corn (Maize)",
        "disease_name": "Common Rust",
        "is_healthy": False,
        "description": (
            "Common rust shows up as small reddish-brown bumps scattered over "
            "both the top and bottom of corn leaves. If you rub them, a "
            "rusty-coloured powder comes off on your finger. Heavy infections "
            "make the leaves look burnt and reduce the size of the cobs."
        ),
        "causes": (
            "A fungus causes this. Its spores blow in on the wind, sometimes "
            "from far away, so it can appear even in a clean field. It does "
            "best in cool, moist weather with high humidity and long periods "
            "of leaf wetness."
        ),
        "treatment": (
            "Light rust on a few leaves usually does not need treating. If "
            "the bumps cover many leaves and the plant has not flowered yet, "
            "spray a suitable fungicide to protect the upper leaves that feed "
            "the cob."
        ),
        "prevention": (
            "Choose corn seed that is described as rust-resistant, since this "
            "is by far the most effective step. Avoid planting too densely, "
            "and do not overdo nitrogen fertiliser, because a thick leafy "
            "canopy stays damp and helps the fungus."
        ),
        "fertilizer_recommendation": (
            "A balanced fertiliser with moderate nitrogen, such as 20-10-10, "
            "split between planting and the knee-high stage, works well. "
            "Avoid piling on extra nitrogen late in the season."
        ),
        "pesticide_recommendation": (
            "Use a fungicide containing azoxystrobin or propiconazole. Spray "
            "at the first sign of the rusty bumps and repeat only if the "
            "label allows and the disease keeps spreading."
        ),
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant_name": "Corn (Maize)",
        "disease_name": "Northern Leaf Blight",
        "is_healthy": False,
        "description": (
            "This disease creates long, grey-green or tan marks shaped like "
            "cigars on the corn leaves. They usually start on the lower "
            "leaves and spread upward. When several marks join, large parts "
            "of the leaf die and the plant fills the cobs poorly."
        ),
        "causes": (
            "A fungus causes it, and it survives on corn leaves and stalks "
            "left in the field from last season. Moderate temperatures, heavy "
            "dew and leaves that stay wet for a long time let it spread "
            "fast."
        ),
        "treatment": (
            "Spray a suitable fungicide as soon as you see the first cigar-"
            "shaped marks, especially if this happens before the plant "
            "flowers. After harvest, plough in or remove the old plant "
            "material."
        ),
        "prevention": (
            "Rotate to a non-corn crop for at least one season. Bury or clear "
            "the leftover stalks and leaves after harvest. Plant hybrid seed "
            "that is labelled resistant to northern leaf blight where you can "
            "get it."
        ),
        "fertilizer_recommendation": (
            "Use a balanced fertiliser such as 20-10-10 with enough "
            "potassium, which helps keep the stalks and leaves strong and "
            "better able to handle infection."
        ),
        "pesticide_recommendation": (
            "Use a fungicide containing a strobilurin (such as azoxystrobin) "
            "or a triazole (such as propiconazole). Apply when the marks "
            "first appear, following the label for timing and repeat "
            "intervals."
        ),
    },
    "Corn_(maize)___healthy": {
        "plant_name": "Corn (Maize)",
        "disease_name": "Healthy",
        "is_healthy": True,
        "description": (
            "This corn leaf looks healthy. It has an even green colour with "
            "no streaks, spots, rusty bumps or dried edges."
        ),
        "causes": (
            "No disease was found, so there is nothing to explain here."
        ),
        "treatment": (
            "No treatment is needed. Carry on with your normal crop care."
        ),
        "prevention": (
            "Rotate corn with other crops instead of planting it in the same "
            "field every year. Clear or plough in old stalks after harvest. "
            "Give the plants enough spacing so air can move between them, and "
            "walk the field regularly to spot problems on the lower leaves "
            "early."
        ),
        "fertilizer_recommendation": (
            "Use a balanced fertiliser such as 20-10-10, split between "
            "planting and the knee-high stage. Corn is a hungry crop, so a "
            "soil test is worth doing to get the nitrogen amount right."
        ),
        "pesticide_recommendation": (
            "No spray is needed for a healthy crop. Keep checking the lower "
            "leaves, since most corn leaf diseases start there first."
        ),
    },

    # ---------------------------------------------------------------- Grape
    "Grape___Black_rot": {
        "plant_name": "Grape",
        "disease_name": "Black Rot",
        "is_healthy": False,
        "description": (
            "Black rot causes small tan spots with dark brown rims on grape "
            "leaves. The real damage is to the fruit: the grapes turn brown, "
            "then shrivel into hard black lumps often called mummies. A bad "
            "attack can ruin most of the bunch."
        ),
        "causes": (
            "A fungus causes it. It survives the winter inside the shrivelled "
            "mummy grapes and old infected canes left on the vine or on the "
            "ground. Warm, wet spring weather releases its spores onto the "
            "new leaves and young fruit."
        ),
        "treatment": (
            "Pick off and destroy every shrivelled mummy grape and cut out "
            "infected canes - this matters more than spraying, because the "
            "mummies are the source of next year's disease. Then spray a "
            "suitable fungicide from the time the buds open."
        ),
        "prevention": (
            "During winter pruning, remove all mummies and dead wood from the "
            "vine and the ground beneath it. Train and prune the vine so the "
            "leaves are open and airy. Water at the base rather than over the "
            "leaves."
        ),
        "fertilizer_recommendation": (
            "Apply a balanced 10-10-10 fertiliser in early spring. Keep "
            "nitrogen moderate, because too much creates a thick leafy canopy "
            "that traps moisture around the fruit."
        ),
        "pesticide_recommendation": (
            "Use a fungicide containing myclobutanil or mancozeb. Start when "
            "the buds open and continue on the label's schedule until the "
            "grapes begin to soften and change colour."
        ),
    },
    "Grape___Esca_(Black_Measles)": {
        "plant_name": "Grape",
        "disease_name": "Esca (Black Measles)",
        "is_healthy": False,
        "description": (
            "Esca is a disease of the wood inside the vine, so it is more "
            "serious than a simple leaf spot. Leaves develop yellow or red "
            "stripes between the veins that later dry out, and the grapes get "
            "small dark spots like measles. Sometimes a whole arm of the vine "
            "suddenly collapses and dies in hot weather."
        ),
        "causes": (
            "Several fungi live inside the older wood of the vine and slowly "
            "block the channels that carry water. They get in through large "
            "pruning cuts and wounds. It usually affects older vines and gets "
            "worse when the vine is stressed by heat or lack of water."
        ),
        "treatment": (
            "There is no spray that cures esca, because the fungi live deep "
            "inside the wood. Cut out and burn the affected arms, cutting "
            "back into clean healthy wood. Badly affected old vines usually "
            "have to be removed and replaced."
        ),
        "prevention": (
            "Prune during dry weather, not before rain, so the cuts can dry "
            "and seal. Keep pruning cuts as small as possible, and paint "
            "large cuts with a wound sealant. Keep the vine well watered and "
            "not stressed, and always clean your pruning tools between "
            "vines."
        ),
        "fertilizer_recommendation": (
            "Feed with a balanced 10-10-10 fertiliser in spring to keep the "
            "vine strong. Good nutrition will not cure esca, but a healthy, "
            "unstressed vine copes with it far better."
        ),
        "pesticide_recommendation": (
            "No ordinary fungicide can cure this disease once it is inside "
            "the wood, so do not waste money spraying the leaves for it. The "
            "only useful chemical step is a protective wound sealant or "
            "paste on fresh pruning cuts."
        ),
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant_name": "Grape",
        "disease_name": "Leaf Blight (Isariopsis Leaf Spot)",
        "is_healthy": False,
        "description": (
            "This fungal disease makes irregular dark brown or black patches "
            "on grape leaves, often with a slightly angular shape. Badly "
            "spotted leaves turn yellow and drop early, which weakens the "
            "vine and leaves the fruit exposed to sunburn."
        ),
        "causes": (
            "A fungus causes it, and it survives on fallen infected leaves. "
            "Rain splashes the spores onto new leaves. Warm, rainy weather "
            "with high humidity and poor air movement inside a crowded canopy "
            "lets it spread."
        ),
        "treatment": (
            "Remove and destroy the badly spotted leaves and any fallen "
            "leaves under the vine. Spray a suitable fungicide to protect the "
            "leaves that are still healthy."
        ),
        "prevention": (
            "Clear fallen leaves from under the vine, as that is where the "
            "fungus waits. Prune and train the vine so the leaves are spread "
            "out and dry quickly after rain. Avoid watering over the top of "
            "the leaves."
        ),
        "fertilizer_recommendation": (
            "A balanced 10-10-10 fertiliser in early spring is suitable. Keep "
            "nitrogen moderate so the canopy does not become too thick and "
            "damp."
        ),
        "pesticide_recommendation": (
            "Use a fungicide containing mancozeb or a copper-based product, "
            "applied when the spots first appear and repeated according to "
            "the label during wet weather."
        ),
    },
    "Grape___healthy": {
        "plant_name": "Grape",
        "disease_name": "Healthy",
        "is_healthy": True,
        "description": (
            "This grape leaf looks healthy. There are no spots, no yellow "
            "stripes and no powdery coating on it."
        ),
        "causes": (
            "No disease was found, so there is nothing to explain here."
        ),
        "treatment": (
            "No treatment is needed. Keep up your normal vineyard care."
        ),
        "prevention": (
            "Prune the vine each winter and remove old wood, mummy grapes and "
            "fallen leaves. Train the shoots so the canopy stays open and the "
            "leaves dry quickly after rain. Water at the base of the vine, and "
            "check the leaves and young bunches regularly."
        ),
        "fertilizer_recommendation": (
            "Apply a balanced 10-10-10 fertiliser in early spring, and add "
            "some potassium while the fruit is developing to help the grapes "
            "ripen well."
        ),
        "pesticide_recommendation": (
            "No spray is needed while the vine is healthy. Continue checking "
            "for early spots, especially after a rainy spell."
        ),
    },

    # --------------------------------------------------------------- Orange
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant_name": "Orange",
        "disease_name": "Huanglongbing (Citrus Greening)",
        "is_healthy": False,
        "description": (
            "Citrus greening is the most serious disease of citrus trees "
            "worldwide. Leaves develop blotchy yellow patches that are "
            "uneven on the two sides of the leaf, the fruit stays small, "
            "lopsided and partly green, and it tastes bitter. Over a few "
            "years the tree slowly declines and dies."
        ),
        "causes": (
            "It is caused by bacteria that live inside the tree's sap "
            "channels. The bacteria are carried from tree to tree by a tiny "
            "insect called the Asian citrus psyllid. It also spreads when "
            "infected budwood is used for grafting new trees."
        ),
        "treatment": (
            "There is no cure for an infected tree. The recommended action is "
            "to remove and destroy the infected tree so it stops being a "
            "source of infection for healthy trees nearby. At the same time, "
            "control the psyllid insects in the area, as they carry the "
            "bacteria."
        ),
        "prevention": (
            "Buy only certified disease-free citrus plants from a reliable "
            "nursery. Check your trees often for the tiny psyllid insects and "
            "for blotchy yellow leaves. Remove infected trees quickly, and "
            "report the disease to your local agriculture office if it is "
            "notifiable in your area."
        ),
        "fertilizer_recommendation": (
            "Use a citrus fertiliser containing micronutrients such as zinc, "
            "iron and manganese, since infected trees suffer badly from "
            "nutrient shortages. Feeding improves how the tree looks and "
            "keeps it producing a little longer, but it does not cure the "
            "disease."
        ),
        "pesticide_recommendation": (
            "No spray kills the bacteria inside the tree, so do not expect a "
            "fungicide or antibiotic spray to cure it. Insecticides are used "
            "only against the psyllid insect that spreads it - horticultural "
            "oil, insecticidal soap or imidacloprid are commonly used. "
            "Imidacloprid is harmful to bees and is restricted in some "
            "countries, so never apply it while the tree is in flower and "
            "check what is allowed where you live."
        ),
    },

    # ---------------------------------------------------------------- Peach
    "Peach___Bacterial_spot": {
        "plant_name": "Peach",
        "disease_name": "Bacterial Spot",
        "is_healthy": False,
        "description": (
            "This bacterial disease makes small dark purple or brown spots on "
            "peach leaves. The centres often fall out, leaving little holes "
            "that look as if someone fired shot through the leaf. The fruit "
            "gets sunken, cracked dark spots on the skin."
        ),
        "causes": (
            "Bacteria cause this, and they survive in cracks and cankers on "
            "the twigs over winter. They are splashed around by rain and "
            "driven into the leaves by wind, so sandy, windy sites suffer "
            "most. Warm, wet weather makes it much worse."
        ),
        "treatment": (
            "Remove badly affected fruit and prune out twigs with cankers "
            "during dry weather. Copper sprays can slow the spread, but they "
            "must be used carefully on peaches because strong copper can "
            "damage the leaves."
        ),
        "prevention": (
            "Plant varieties described as resistant to bacterial spot, as "
            "this makes the biggest difference. Provide a windbreak if your "
            "site is exposed, since wind-driven rain pushes the bacteria into "
            "the leaves. Avoid overhead watering and do not prune while the "
            "tree is wet."
        ),
        "fertilizer_recommendation": (
            "Use a balanced 10-10-10 fertiliser in early spring and keep the "
            "amount moderate. Too much nitrogen makes soft leafy growth that "
            "the bacteria attack more easily."
        ),
        "pesticide_recommendation": (
            "Use a copper-based spray while the tree is still bare in late "
            "winter, and low-rate copper during the season if needed. Follow "
            "the label rates exactly, because too much copper burns peach "
            "leaves. Be realistic: sprays only slow this disease down, so "
            "choosing a resistant variety matters far more than spraying."
        ),
    },
    "Peach___healthy": {
        "plant_name": "Peach",
        "disease_name": "Healthy",
        "is_healthy": True,
        "description": (
            "This peach leaf looks healthy. There are no dark spots, no shot "
            "holes and no curling or thickening of the leaf."
        ),
        "causes": (
            "No disease was found, so there is nothing to explain here."
        ),
        "treatment": (
            "No treatment is needed. Continue with your usual care."
        ),
        "prevention": (
            "Prune the tree each year so the centre stays open and airy, and "
            "do your pruning in dry weather. Clear fallen leaves and old "
            "fruit from under the tree. Water at the base, not over the "
            "leaves, and check the new leaves each spring for early spots or "
            "curling."
        ),
        "fertilizer_recommendation": (
            "Apply a balanced 10-10-10 fertiliser in early spring. Peaches "
            "respond well to steady moderate feeding rather than one heavy "
            "dose."
        ),
        "pesticide_recommendation": (
            "No spray is needed while the tree is healthy. A single dormant "
            "spray in late winter is a common preventive habit in areas where "
            "peach leaf curl is a known local problem."
        ),
    },

    # ----------------------------------------------------------- Bell Pepper
    "Pepper,_bell___Bacterial_spot": {
        "plant_name": "Bell Pepper",
        "disease_name": "Bacterial Spot",
        "is_healthy": False,
        "description": (
            "This bacterial disease makes small water-soaked spots on pepper "
            "leaves that turn brown with a yellow ring around them. Affected "
            "leaves drop off, leaving the plant bare, and the peppers develop "
            "raised rough scabby patches that make them unsellable."
        ),
        "causes": (
            "Bacteria cause this. They arrive on infected seed or young "
            "plants, then spread by splashing water, rain and by people and "
            "tools touching wet plants. Warm, humid, rainy weather is ideal "
            "for them."
        ),
        "treatment": (
            "Remove the worst affected leaves and any fallen leaves around "
            "the plant. Stop watering over the top of the plants. Copper "
            "sprays can slow the spread but will not cure plants that are "
            "already badly infected."
        ),
        "prevention": (
            "Start with certified disease-free seed or seedlings, since the "
            "disease often arrives that way. Do not plant peppers or tomatoes "
            "in the same spot two years running. Water at the base of the "
            "plant, and never work among the plants while the leaves are wet, "
            "because that spreads the bacteria on your hands and clothes."
        ),
        "fertilizer_recommendation": (
            "Use a balanced 10-10-10 fertiliser and avoid heavy nitrogen "
            "feeding, which produces soft leaves that get infected more "
            "easily. Add potassium when the fruit starts to set."
        ),
        "pesticide_recommendation": (
            "Use a copper-based spray, often mixed with mancozeb where that "
            "combination is permitted, applied about every 7 days during warm "
            "wet weather. Remember these are protective sprays - they shield "
            "healthy leaves rather than cure sick ones."
        ),
    },
    "Pepper,_bell___healthy": {
        "plant_name": "Bell Pepper",
        "disease_name": "Healthy",
        "is_healthy": True,
        "description": (
            "This bell pepper leaf looks healthy. The colour is good and "
            "there are no spots, holes, curling or yellowing."
        ),
        "causes": (
            "No disease was found, so there is nothing to explain here."
        ),
        "treatment": (
            "No treatment is needed. Keep up your current care routine."
        ),
        "prevention": (
            "Water at the base of the plant and keep the leaves dry. Leave "
            "enough space between plants so air can move around them. Stake "
            "or support the plants to keep leaves and fruit off the soil, and "
            "rotate where you plant peppers each year."
        ),
        "fertilizer_recommendation": (
            "Use a balanced 10-10-10 fertiliser at planting, then switch to a "
            "feed higher in potassium once the flowers and fruit appear, "
            "which helps the peppers develop well."
        ),
        "pesticide_recommendation": (
            "No spray is needed for a healthy plant. Keep checking the "
            "undersides of the leaves, where pests usually show up first."
        ),
    },

    # --------------------------------------------------------------- Potato
    "Potato___Early_blight": {
        "plant_name": "Potato",
        "disease_name": "Early Blight",
        "is_healthy": False,
        "description": (
            "Early blight makes dark brown spots on the older, lower leaves "
            "first. The spots have rings inside them, a bit like a target or "
            "a tree stump. A yellow halo often surrounds each spot, and badly "
            "affected leaves dry up and fall, which means smaller potatoes."
        ),
        "causes": (
            "A fungus causes this. It survives in old potato and tomato "
            "debris in the soil, and rain splashes it up onto the lowest "
            "leaves. Warm weather that switches between wet and dry suits it, "
            "and plants that are short of food or water get hit hardest."
        ),
        "treatment": (
            "Pick off and destroy the badly spotted lower leaves. Put a mulch "
            "such as straw on the soil to stop rain splashing spores upward. "
            "Spray a suitable fungicide as soon as the spots appear."
        ),
        "prevention": (
            "Rotate your crops so potatoes are not grown in the same bed for "
            "2 to 3 years. Clear away old potato and tomato plants after "
            "harvest. Space the plants well, water at the base, and keep the "
            "plants properly fed, since weak plants suffer far more."
        ),
        "fertilizer_recommendation": (
            "Use a balanced fertiliser such as 14-14-14 with plenty of "
            "potassium. Well-fed plants resist early blight much better, so "
            "do not let the crop go hungry - this is one disease where good "
            "feeding genuinely helps."
        ),
        "pesticide_recommendation": (
            "Use a fungicide containing chlorothalonil or mancozeb. Spray "
            "every 7 to 10 days once the first spots appear, making sure to "
            "cover the lower leaves where the disease starts."
        ),
    },
    "Potato___Late_blight": {
        "plant_name": "Potato",
        "disease_name": "Late Blight",
        "is_healthy": False,
        "description": (
            "Late blight is the most destructive potato disease - it is the "
            "one that caused the Irish potato famine. It makes dark, "
            "water-soaked, greasy-looking patches on the leaves and stems, "
            "often with a white fuzzy growth underneath in damp weather. In "
            "the right conditions it can destroy a whole crop within days."
        ),
        "causes": (
            "It is caused by a fungus-like organism that spreads extremely "
            "fast in cool, wet, humid weather. It survives in infected seed "
            "potatoes, in leftover tubers in the soil, and in volunteer "
            "potato plants that come up on their own."
        ),
        "treatment": (
            "Act immediately - do not wait. Pull out and destroy infected "
            "plants (burn or bin them, do not compost them) to stop the "
            "spread to the rest of the crop. Spray the remaining healthy "
            "plants with a suitable fungicide without delay."
        ),
        "prevention": (
            "Plant certified disease-free seed potatoes. Never leave old "
            "tubers or volunteer potato plants in the field. Earth up the "
            "rows well so the tubers are covered and spores cannot wash down "
            "to them. Water at the base, ensure good drainage, and space "
            "plants for airflow."
        ),
        "fertilizer_recommendation": (
            "Use a balanced 14-14-14 fertiliser and avoid excess nitrogen, "
            "which creates dense soft foliage that stays wet and catches the "
            "disease more easily. Adequate potassium supports stronger "
            "plants."
        ),
        "pesticide_recommendation": (
            "Use a fungicide combining metalaxyl and mancozeb, or "
            "chlorothalonil. In wet weather these should be applied "
            "preventively every 5 to 7 days, because once the disease takes "
            "hold it is very hard to stop."
        ),
    },
    "Potato___healthy": {
        "plant_name": "Potato",
        "disease_name": "Healthy",
        "is_healthy": True,
        "description": (
            "This potato leaf looks healthy. There are no dark patches, no "
            "ringed spots and no yellowing or wilting."
        ),
        "causes": (
            "No disease was found, so there is nothing to explain here."
        ),
        "treatment": (
            "No treatment is needed. Continue with your normal care."
        ),
        "prevention": (
            "Always start with certified seed potatoes. Rotate the crop so "
            "potatoes do not return to the same bed for 2 to 3 years. Earth "
            "up the rows to protect the tubers, water at the base, and check "
            "the plants often - especially during cool wet spells when blight "
            "can appear suddenly."
        ),
        "fertilizer_recommendation": (
            "Use a balanced 14-14-14 fertiliser, split between planting time "
            "and when you earth up the rows. Potatoes particularly appreciate "
            "potassium, which helps the tubers develop."
        ),
        "pesticide_recommendation": (
            "No spray is needed while the crop is healthy. Watch the weather, "
            "though - a cool, wet, humid spell is exactly when blight arrives "
            "and preventive action may become worthwhile."
        ),
    },

    # ------------------------------------------------------------ Raspberry
    "Raspberry___healthy": {
        "plant_name": "Raspberry",
        "disease_name": "Healthy",
        "is_healthy": True,
        "description": (
            "This raspberry leaf looks healthy. The leaf colour is even, with "
            "no spots, rusty patches or dried edges."
        ),
        "causes": (
            "No disease was found, so there is nothing to explain here."
        ),
        "treatment": (
            "No treatment is needed. Keep doing what you are doing."
        ),
        "prevention": (
            "Cut out the old canes that have already fruited, right down to "
            "the ground, and remove them - this is the single most useful job "
            "for raspberry health. Keep the rows narrow and tie the canes to "
            "supports so air moves freely. Water at the base and keep the bed "
            "free of weeds."
        ),
        "fertilizer_recommendation": (
            "Apply a balanced 10-10-10 fertiliser in early spring as the new "
            "canes start growing. A mulch of well-rotted compost around the "
            "base feeds the plants slowly and keeps the roots cool."
        ),
        "pesticide_recommendation": (
            "No spray is needed for healthy canes. Good pruning and removing "
            "old canes does more to prevent problems than any spray."
        ),
    },

    # -------------------------------------------------------------- Soybean
    "Soybean___healthy": {
        "plant_name": "Soybean",
        "disease_name": "Healthy",
        "is_healthy": True,
        "description": (
            "This soybean leaf looks healthy. It has a good even green colour "
            "with no spots, yellow patches or chewed edges."
        ),
        "causes": (
            "No disease was found, so there is nothing to explain here."
        ),
        "treatment": (
            "No treatment is needed. Continue your normal crop care."
        ),
        "prevention": (
            "Rotate soybeans with a different crop such as corn rather than "
            "planting them in the same field repeatedly. Clear or plough in "
            "old plant debris after harvest. Use good quality seed, avoid "
            "planting too densely, and walk the field regularly to check the "
            "lower leaves."
        ),
        "fertilizer_recommendation": (
            "Soybeans make much of their own nitrogen through their roots, so "
            "they need little nitrogen fertiliser. Focus on phosphorus and "
            "potassium based on a soil test, and make sure the seed is "
            "properly treated with rhizobium bacteria so the roots can fix "
            "nitrogen."
        ),
        "pesticide_recommendation": (
            "No spray is needed for a healthy crop. Keep scouting the field, "
            "as it is much cheaper to act early than to treat a widespread "
            "problem."
        ),
    },

    # --------------------------------------------------------------- Squash
    "Squash___Powdery_mildew": {
        "plant_name": "Squash",
        "disease_name": "Powdery Mildew",
        "is_healthy": False,
        "description": (
            "Powdery mildew covers squash leaves with white powdery patches "
            "that look like spilled flour. It usually starts on the older "
            "lower leaves and spreads upward. Affected leaves turn yellow, "
            "then brown and crispy, and the fruit ripens poorly or gets "
            "sunburnt once the leaves die back."
        ),
        "causes": (
            "A fungus causes it. Unlike most leaf diseases it does not need "
            "wet leaves to infect - warm dry days followed by humid nights "
            "suit it best. Crowded plants and shady spots with little air "
            "movement keep the air around the leaves humid, which is what "
            "lets it spread."
        ),
        "treatment": (
            "Remove and destroy the worst affected lower leaves to slow the "
            "spread and let more air in. Spray the remaining leaves, covering "
            "both the top and the underside. Potassium bicarbonate or "
            "sulphur sprays work well if you catch it early, while it is "
            "still only on a few leaves."
        ),
        "prevention": (
            "Give the plants plenty of space when planting, since crowding is "
            "the main cause. Grow varieties labelled mildew-resistant where "
            "you can. Water the soil, not the leaves, and water in the "
            "morning so any moisture dries during the day. Remove old squash "
            "plants at the end of the season."
        ),
        "fertilizer_recommendation": (
            "Use a balanced 10-10-10 fertiliser and keep nitrogen moderate, "
            "because lush soft leaves get mildew fastest. Once fruit begins "
            "to form, a feed higher in potassium supports the developing "
            "squash."
        ),
        "pesticide_recommendation": (
            "Use potassium bicarbonate, sulphur, neem oil or a fungicide "
            "containing myclobutanil. Spray at the first white patches and "
            "repeat every 7 to 14 days. Do not apply sulphur or oils in very "
            "hot weather, as they can burn the leaves."
        ),
    },

    # ----------------------------------------------------------- Strawberry
    "Strawberry___Leaf_scorch": {
        "plant_name": "Strawberry",
        "disease_name": "Leaf Scorch",
        "is_healthy": False,
        "description": (
            "Leaf scorch starts as small dark purple spots on strawberry "
            "leaves. As more appear they join together, and the leaf edges "
            "look burnt or scorched, as if they had been held near a flame. "
            "Badly affected plants produce fewer and smaller berries."
        ),
        "causes": (
            "A fungus causes this. It survives on old infected leaves left in "
            "the bed from last season. Warm, wet weather and thick, crowded "
            "plantings with poor air movement let it spread easily."
        ),
        "treatment": (
            "Cut off and destroy the badly scorched leaves, especially after "
            "harvest when you tidy the bed. Thin out crowded plants so air "
            "can move through, and spray a suitable fungicide if the disease "
            "keeps spreading."
        ),
        "prevention": (
            "Space the plants properly and thin out the runners so the bed "
            "does not become a dense mat. Remove old leaves after harvest "
            "each year. Water at the base in the morning rather than over the "
            "leaves, and choose resistant varieties when replanting."
        ),
        "fertilizer_recommendation": (
            "Apply a balanced 10-10-10 fertiliser after harvest when you cut "
            "back the old leaves, to help the plants push out strong new "
            "growth for next season."
        ),
        "pesticide_recommendation": (
            "Use a fungicide containing captan or myclobutanil, applied every "
            "7 to 14 days while the disease is actively spreading. Follow the "
            "label carefully for how long to wait before picking berries."
        ),
    },
    "Strawberry___healthy": {
        "plant_name": "Strawberry",
        "disease_name": "Healthy",
        "is_healthy": True,
        "description": (
            "This strawberry leaf looks healthy. There are no purple spots, "
            "no scorched edges and no powdery coating."
        ),
        "causes": (
            "No disease was found, so there is nothing to explain here."
        ),
        "treatment": (
            "No treatment is needed. Carry on with your usual care."
        ),
        "prevention": (
            "Keep the plants properly spaced and thin out excess runners so "
            "the bed stays open. Remove old leaves after harvest. Use a straw "
            "mulch to keep the berries off the soil, water at the base in the "
            "morning, and check the leaves regularly."
        ),
        "fertilizer_recommendation": (
            "Apply a balanced 10-10-10 fertiliser after harvest, with light "
            "feeding during the growing season. Avoid heavy nitrogen while "
            "the fruit is forming, as it gives you leaves instead of "
            "berries."
        ),
        "pesticide_recommendation": (
            "No spray is needed while the plants are healthy. Netting against "
            "birds and a straw mulch to keep fruit clean are usually more "
            "useful than chemicals."
        ),
    },

    # --------------------------------------------------------------- Tomato
    "Tomato___Bacterial_spot": {
        "plant_name": "Tomato",
        "disease_name": "Bacterial Spot",
        "is_healthy": False,
        "description": (
            "This bacterial disease makes many small dark spots on tomato "
            "leaves, often with a yellow edge. The spots also appear on stems "
            "and fruit, where they become raised and scabby. Badly affected "
            "leaves drop off and the exposed fruit gets sunburnt."
        ),
        "causes": (
            "Bacteria cause this. They usually arrive on infected seed or "
            "young plants, then spread by splashing rain, overhead watering, "
            "and hands and tools touching wet plants. Warm, humid, wet "
            "weather is perfect for them."
        ),
        "treatment": (
            "Remove the worst affected leaves and clear any fallen leaves "
            "from around the plants. Switch to watering at the base only. "
            "Copper sprays help protect the healthy leaves but will not cure "
            "leaves that are already spotted."
        ),
        "prevention": (
            "Use certified disease-free seed or seedlings. Rotate so tomatoes "
            "and peppers do not grow in the same spot each year. Stake the "
            "plants to lift leaves off the soil, water at the base, and never "
            "handle the plants while they are wet."
        ),
        "fertilizer_recommendation": (
            "Use a balanced 10-10-10 fertiliser and keep nitrogen moderate, "
            "since soft lush growth gets infected more easily. Increase "
            "potassium once the fruit starts to form."
        ),
        "pesticide_recommendation": (
            "Use a copper-based spray, sometimes combined with mancozeb, "
            "applied about every 7 days during warm wet weather. These sprays "
            "protect healthy leaves rather than cure infected ones."
        ),
    },
    "Tomato___Early_blight": {
        "plant_name": "Tomato",
        "disease_name": "Early Blight",
        "is_healthy": False,
        "description": (
            "Early blight makes dark brown spots with rings inside them, like "
            "a target, on the older lower leaves first. A yellow area usually "
            "surrounds each spot. The damage moves steadily up the plant, and "
            "dark sunken patches can also form on the stem and the fruit near "
            "the stalk."
        ),
        "causes": (
            "A fungus causes it. It survives in old tomato and potato debris "
            "in the soil, and rain or watering splashes it onto the lowest "
            "leaves. Warm humid weather and plants that are short of "
            "nutrients or water are hit hardest."
        ),
        "treatment": (
            "Pick off and destroy the affected lower leaves as soon as you "
            "see them. Put a straw or plastic mulch on the soil to stop rain "
            "splashing spores upward. Spray the remaining leaves with a "
            "suitable fungicide."
        ),
        "prevention": (
            "Rotate crops so tomatoes do not follow tomatoes or potatoes. "
            "Stake or cage the plants so air flows and the lower leaves stay "
            "clear of the soil. Remove the lowest leaves near the ground, "
            "water at the base, and clear away old plants at the end of the "
            "season."
        ),
        "fertilizer_recommendation": (
            "Use a balanced 10-10-10 fertiliser and keep the plants "
            "consistently fed, since hungry plants get early blight much "
            "worse. Adding calcium also helps the plant build tougher "
            "tissue."
        ),
        "pesticide_recommendation": (
            "Use a fungicide containing chlorothalonil or mancozeb, applied "
            "every 7 to 10 days once the first spots appear. Make sure to "
            "cover the lower leaves where the disease begins."
        ),
    },
    "Tomato___Late_blight": {
        "plant_name": "Tomato",
        "disease_name": "Late Blight",
        "is_healthy": False,
        "description": (
            "Late blight is the most dangerous tomato disease. It makes "
            "large, irregular, water-soaked patches that look greasy on the "
            "leaves and stems, often with white fuzzy growth on the "
            "underside in damp weather. The fruit develops firm brown "
            "blotches. A whole plant can collapse within a few days."
        ),
        "causes": (
            "It is caused by a fungus-like organism that spreads extremely "
            "fast in cool, wet, humid weather. Its spores blow in on the wind "
            "and can also come from nearby infected potato plants or "
            "leftover tubers."
        ),
        "treatment": (
            "Act immediately. Pull out and destroy infected plants - burn or "
            "bin them, never compost them - because one infected plant will "
            "infect the whole row. Spray the remaining healthy plants with a "
            "suitable fungicide straight away."
        ),
        "prevention": (
            "Space plants widely and stake them so air moves freely. Water at "
            "the base early in the day so leaves dry fast. Do not grow "
            "tomatoes right next to potatoes. Remove volunteer tomato and "
            "potato plants, and choose blight-resistant varieties where "
            "available."
        ),
        "fertilizer_recommendation": (
            "Use a balanced 10-10-10 fertiliser and avoid too much nitrogen, "
            "which creates dense soft leaves that stay wet. Extra potassium "
            "helps the plant stay sturdy."
        ),
        "pesticide_recommendation": (
            "Use a fungicide containing chlorothalonil, or a metalaxyl and "
            "mancozeb combination. In cool wet weather apply preventively "
            "every 5 to 7 days, as treatment after the disease is "
            "established rarely saves the plant."
        ),
    },
    "Tomato___Leaf_Mold": {
        "plant_name": "Tomato",
        "disease_name": "Leaf Mold",
        "is_healthy": False,
        "description": (
            "Leaf mold shows as pale green or yellow patches on the upper "
            "side of tomato leaves, with a fuzzy olive-green to grey-purple "
            "mould growing on the underside. It is especially common in "
            "greenhouses and polytunnels. Affected leaves dry up and fall, "
            "reducing the crop."
        ),
        "causes": (
            "A fungus causes it, and it needs high humidity - above about 85 "
            "per cent - to get going. Poor ventilation, crowded plants and "
            "leaves that stay damp overnight are the main reasons it "
            "appears, which is why greenhouse tomatoes suffer most."
        ),
        "treatment": (
            "Improve the airflow immediately: open vents, run a fan, and "
            "remove some lower leaves to let air circulate. Take off the "
            "badly affected leaves. Then spray a suitable fungicide, "
            "covering the underside of the leaves where the mould grows."
        ),
        "prevention": (
            "Lowering the humidity is the key step - ventilate the "
            "greenhouse well, especially in the evening, and avoid letting "
            "the air get damp and still overnight. Space plants generously, "
            "prune lower leaves, water the soil not the leaves, and grow "
            "resistant varieties if you can."
        ),
        "fertilizer_recommendation": (
            "A balanced 10-10-10 fertiliser is fine. Do not overfeed with "
            "nitrogen, because a thick leafy plant traps humid air around the "
            "leaves, which is exactly what this disease wants."
        ),
        "pesticide_recommendation": (
            "Use a fungicide containing chlorothalonil or a copper-based "
            "product, applied every 7 to 10 days while humidity stays high. "
            "Spray the undersides of the leaves properly, as that is where "
            "the mould lives."
        ),
    },
    "Tomato___Septoria_leaf_spot": {
        "plant_name": "Tomato",
        "disease_name": "Septoria Leaf Spot",
        "is_healthy": False,
        "description": (
            "This disease covers tomato leaves with many small round spots "
            "that have dark edges and pale grey or tan centres. There are "
            "usually far more spots than with early blight, and no target "
            "rings. It starts on the lowest leaves and works upward, "
            "stripping the plant of foliage."
        ),
        "causes": (
            "A fungus causes it. It survives on old infected tomato debris "
            "and on some weeds. Rain and overhead watering splash the spores "
            "onto the lower leaves, and warm humid weather makes it spread "
            "quickly."
        ),
        "treatment": (
            "Remove the spotted lower leaves as soon as you notice them and "
            "destroy them. Add a mulch to stop soil splashing onto the "
            "leaves, and spray a suitable fungicide to protect the leaves "
            "higher up."
        ),
        "prevention": (
            "Rotate crops and clear away all old tomato plants at the end of "
            "the season. Keep weeds down, as some of them harbour the fungus. "
            "Stake the plants, remove the lowest leaves, mulch the soil, and "
            "always water at the base."
        ),
        "fertilizer_recommendation": (
            "A balanced 10-10-10 fertiliser keeps the plant growing steadily. "
            "Moderate feeding is best - very lush plants hold moisture "
            "around the leaves and lose more foliage to this disease."
        ),
        "pesticide_recommendation": (
            "Use a fungicide containing chlorothalonil or a copper-based "
            "product, applied every 7 to 10 days once the spots appear, "
            "concentrating on protecting the healthy upper leaves."
        ),
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant_name": "Tomato",
        "disease_name": "Spider Mites (Two-Spotted Spider Mite)",
        "is_healthy": False,
        "description": (
            "This is damage from a tiny pest, not a disease. Spider mites are "
            "almost too small to see, but they suck sap from the underside of "
            "the leaves. The leaves develop fine yellow or white speckles, "
            "then turn bronze and dry up. In a bad attack you can see fine "
            "webbing between the leaves and stems."
        ),
        "causes": (
            "Spider mites multiply very fast in hot, dry, dusty conditions - "
            "the opposite of most tomato diseases. Plants that are short of "
            "water are especially vulnerable. Mites often arrive on new "
            "plants or blow in on the wind, and some insecticides make things "
            "worse by killing the natural predators that eat mites."
        ),
        "treatment": (
            "Spray the plants forcefully with plain water, aiming at the "
            "underside of the leaves - this alone knocks off many mites. "
            "Remove the worst affected leaves. Then treat with insecticidal "
            "soap, neem oil or a proper miticide, making sure to cover the "
            "leaf undersides where the mites live."
        ),
        "prevention": (
            "Keep the plants properly watered, as drought-stressed plants are "
            "the ones that get infested. Hose down dusty leaves now and then. "
            "Check leaf undersides regularly with a magnifying glass. Avoid "
            "spraying broad insecticides unnecessarily, because they kill the "
            "helpful predator insects that keep mites in check."
        ),
        "fertilizer_recommendation": (
            "Use a balanced 10-10-10 fertiliser and keep nitrogen moderate. "
            "Heavy nitrogen feeding actually encourages spider mites to breed "
            "faster, so steady moderate feeding is better here."
        ),
        "pesticide_recommendation": (
            "Ordinary fungicides do nothing against mites, so do not use "
            "them. Use insecticidal soap, neem oil, horticultural oil, or a "
            "specific miticide such as abamectin. Spray the undersides "
            "thoroughly and repeat after 5 to 7 days to catch newly hatched "
            "mites."
        ),
    },
    "Tomato___Target_Spot": {
        "plant_name": "Tomato",
        "disease_name": "Target Spot",
        "is_healthy": False,
        "description": (
            "Target spot makes small brown spots on tomato leaves that grow "
            "larger and develop light-coloured centres with faint rings, "
            "giving them a target-like look. The spots also appear on stems "
            "and on the fruit, where they become sunken pits that spoil the "
            "tomatoes."
        ),
        "causes": (
            "A fungus causes it. It survives on old infected plant debris, "
            "and spreads in warm, wet, humid weather, particularly where "
            "plants are dense and the leaves stay wet for long periods."
        ),
        "treatment": (
            "Remove and destroy the affected leaves and any fruit with sunken "
            "spots. Improve the air movement by pruning crowded growth, and "
            "spray a suitable fungicide to protect the rest of the plant."
        ),
        "prevention": (
            "Rotate crops and clear old tomato plants away after harvest. "
            "Space and stake the plants so air moves between them, prune off "
            "the lowest leaves, water at the base, and avoid working among "
            "wet plants."
        ),
        "fertilizer_recommendation": (
            "Use a balanced 10-10-10 fertiliser with adequate potassium to "
            "keep the plant strong. Avoid excessive nitrogen, which makes the "
            "canopy dense and damp."
        ),
        "pesticide_recommendation": (
            "Use a fungicide containing chlorothalonil, mancozeb or "
            "azoxystrobin, applied every 7 to 10 days during warm wet "
            "weather, covering both sides of the leaves."
        ),
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant_name": "Tomato",
        "disease_name": "Tomato Yellow Leaf Curl Virus",
        "is_healthy": False,
        "description": (
            "This is a virus disease, and it is one of the most damaging "
            "problems tomatoes face. The leaves curl upward, become small "
            "and crumpled, and turn yellow at the edges. The whole plant "
            "stays stunted and produces very few tomatoes, especially if it "
            "was infected while young."
        ),
        "causes": (
            "A virus causes it, and it is carried from plant to plant by tiny "
            "white flying insects called whiteflies. The virus does not "
            "spread by touch, tools or water - the insect is the only "
            "important route, which is why controlling whiteflies is the "
            "whole battle."
        ),
        "treatment": (
            "There is no cure - once a plant has the virus it will not "
            "recover. Pull out badly affected plants and destroy them so they "
            "stop being a source of virus for the whiteflies to pick up. "
            "Then focus on reducing the whitefly numbers around your "
            "remaining plants."
        ),
        "prevention": (
            "Grow tomato varieties described as TYLCV-resistant, which is the "
            "most effective step available. Use fine insect netting over young "
            "plants, and lay reflective silver mulch, which confuses "
            "whiteflies. Put up yellow sticky traps to monitor them, and "
            "control whiteflies early before numbers build up."
        ),
        "fertilizer_recommendation": (
            "Use a balanced 10-10-10 fertiliser to keep healthy plants "
            "growing well. Be clear, though: fertiliser cannot cure or "
            "reverse a virus infection - it only helps uninfected plants stay "
            "strong."
        ),
        "pesticide_recommendation": (
            "No fungicide works on a virus, so do not spray one for this. "
            "Insecticides are used only against the whitefly that carries it "
            "- insecticidal soap or neem oil are the safest choices for a "
            "home garden. Spray the leaf undersides where whiteflies gather. "
            "Stronger products such as imidacloprid are harmful to bees and "
            "restricted in some countries, so check local rules and never "
            "spray them on open flowers."
        ),
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant_name": "Tomato",
        "disease_name": "Tomato Mosaic Virus",
        "is_healthy": False,
        "description": (
            "This virus gives the leaves a mottled pattern of light and dark "
            "green patches, like a mosaic. Leaves may also be narrow, "
            "wrinkled or fern-like. Plants grow slowly, and the fruit can "
            "ripen unevenly with brown patches inside."
        ),
        "causes": (
            "A virus causes it, and unlike most tomato viruses it spreads "
            "mainly by touch rather than by insects. It passes on hands, "
            "clothes, tools, and infected seed, and it is unusually tough - "
            "it can survive in dried plant debris and even in tobacco "
            "products for a long time."
        ),
        "treatment": (
            "There is no cure. Remove and destroy infected plants so they do "
            "not spread the virus to healthy ones. Then wash your hands "
            "thoroughly with soap and disinfect every tool, stake and tray "
            "you touched, because the virus survives on surfaces."
        ),
        "prevention": (
            "Use certified virus-free seed and resistant varieties. Wash your "
            "hands before handling tomato plants, especially if you smoke, "
            "since the virus can be present in tobacco. Disinfect tools, "
            "stakes and trays between uses, and do not save seed from "
            "infected plants."
        ),
        "fertilizer_recommendation": (
            "A balanced 10-10-10 fertiliser keeps healthy plants growing "
            "strongly. Remember that no amount of feeding will cure a plant "
            "that already has the virus."
        ),
        "pesticide_recommendation": (
            "No fungicide or insecticide can cure or prevent this virus, "
            "because it spreads by touch rather than by insects. Strict "
            "hygiene - clean hands, clean tools, clean seed - is the only "
            "real control, so spraying is not the answer here."
        ),
    },
    "Tomato___healthy": {
        "plant_name": "Tomato",
        "disease_name": "Healthy",
        "is_healthy": True,
        "description": (
            "This tomato leaf looks healthy. The colour is even, the shape is "
            "normal, and there are no spots, curling, mottling or speckles."
        ),
        "causes": (
            "No disease was found, so there is nothing to explain here."
        ),
        "treatment": (
            "No treatment is needed. Keep up your current care routine."
        ),
        "prevention": (
            "Stake or cage the plants so air flows around them and the leaves "
            "stay off the soil. Water at the base early in the day. Put down "
            "a mulch to stop soil splashing onto the lower leaves. Rotate "
            "where you plant tomatoes each year, and check the leaves - "
            "including the undersides - about twice a week."
        ),
        "fertilizer_recommendation": (
            "Use a balanced 10-10-10 fertiliser at planting, then switch to a "
            "feed lower in nitrogen and higher in potassium once flowering "
            "starts, so the plant puts its energy into fruit rather than "
            "leaves."
        ),
        "pesticide_recommendation": (
            "No spray is needed while the plant is healthy. Regular checking "
            "of the leaf undersides is the best habit, as most tomato "
            "problems show up there first."
        ),
    },
}


def _humanize(text: str) -> str:
    """Turn 'Northern_Leaf_Blight' or 'bell' into 'Northern Leaf Blight'."""
    return text.replace('_', ' ').replace('-', ' ').strip().title()


def _generic_entry(class_name: str) -> dict:
    """
    Safety net for any class name that is not listed in DISEASE_INFO.

    All 38 classes of the current trained model ARE listed above, so this
    should not normally be reached. It stays in place so the app still
    returns a complete, sensible result (instead of crashing) if the model
    is ever retrained with extra or renamed classes.
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
            "description": (
                f"This {plant_name.lower()} leaf looks healthy, with no "
                f"visible signs of disease."
            ),
            "causes": "No disease was found, so there is nothing to explain here.",
            "treatment": "No treatment is needed. Continue your usual care.",
            "prevention": (
                "Water regularly at the base of the plant, feed it in a "
                "balanced way, give it enough space for air to move around, "
                "and check the leaves often so you catch any problem early."
            ),
            "fertilizer_recommendation": (
                "A balanced 10-10-10 fertiliser following the normal schedule "
                "for this crop is usually enough."
            ),
            "pesticide_recommendation": (
                "No spray is needed for a healthy plant. Just keep checking "
                "it regularly."
            ),
        }

    return {
        "plant_name": plant_name,
        "disease_name": disease_name,
        "is_healthy": False,
        "description": (
            f"The model thinks this {plant_name.lower()} leaf shows signs of "
            f"{disease_name}. Detailed notes for this particular problem have "
            f"not been added to the guide yet."
        ),
        "causes": (
            "Detailed information for this problem is not available yet. Most "
            "leaf diseases are caused by fungi or bacteria that spread "
            "through water, wind, or dirty tools."
        ),
        "treatment": (
            "Keep the affected plant away from healthy ones, remove the "
            "damaged leaves, and ask a local agricultural office or a plant "
            "expert for advice on the right treatment."
        ),
        "prevention": (
            "General good practice: rotate your crops, water at the base "
            "instead of over the leaves, leave space between plants for air "
            "to move, and clean your tools between plants."
        ),
        "fertilizer_recommendation": (
            "A balanced 10-10-10 fertiliser is a safe choice unless a soil "
            "test shows that something specific is missing."
        ),
        "pesticide_recommendation": (
            "Find out exactly what the problem is before buying any spray. A "
            "local agricultural office can identify it and tell you the right "
            "product to use."
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