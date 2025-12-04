import re 
import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plt  
import streamlit as st  

def clean_ingredients(ingredients_text):
    """
    Convert the ingredients text into a clean list.
    Uses: string methods, lists, loops
    """
   
    raw_parts = re.split(r",|\n", ingredients_text)
    clean_list = []

    for item in raw_parts:
        word = item.strip().lower()
        if word:
            clean_list.append(word)

    return clean_list


def classify_processing_level(ingredients_text):
    """
    Very simple rule-based classifier.
    NOT scientifically perfect – only for project demo.
    Uses: lists, sets, loops, if-else
    """
    ingredients = clean_ingredients(ingredients_text)

  
    ultra_processed_keywords = [
        "emulsifier", "stabilizer", "stabiliser", "preservative",
        "acidity regulator", "flavour", "flavor", "artificial",
        "colour", "color", "maltodextrin", "monosodium glutamate",
        "msg", "hydrogenated", "shortening", "sweetener", "corn syrup"
    ]

   
    marker_count = 0
    for ing in ingredients:
        for keyword in ultra_processed_keywords:
            if keyword in ing:
                marker_count += 1

    
    if len(ingredients) <= 3 and marker_count == 0:
        level = "Minimally processed"
        reason = "Few ingredients and no typical additives were detected."
    elif marker_count <= 2:
        level = "Processed"
        reason = "Some processing/seasoning is likely but only few additives were detected."
    else:
        level = "Highly processed"
        reason = "Many ingredients and several additive-related words were detected."

    return level, reason


def guess_product_category(ingredients_text):
    """
    Guess product category based on some simple words.
    """
    text = ingredients_text.lower()

    if "chips" in text or "fried" in text or "namkeen" in text or "snack" in text:
        return "snack"
    if "biscuit" in text or "cookie" in text or "cake" in text:
        return "bakery"
    if "cola" in text or "soft drink" in text or "juice" in text or "drink" in text:
        return "drink"
    if "noodle" in text or "pasta" in text or "maggi" in text:
        return "instant_meal"
    return "other"


def suggest_healthier_alternatives(category, processing_level):
    """
    Suggest very simple alternatives based on category.
    Uses: dictionary, lists, if-else
    """
    alternatives = {
        "snack": [
            "Roasted chana (gram) instead of fried chips",
            "Homemade popcorn with little oil",
            "Mixed nuts and seeds in small quantity"
        ],
        "bakery": [
            "Whole wheat homemade cookies with less sugar",
            "Plain rusk / khakhra instead of cream biscuits",
            "Fruit with a handful of nuts instead of cake"
        ],
        "drink": [
            "Plain water or infused water (lemon, mint)",
            "Unsweetened coconut water",
            "Homemade lime water with less sugar"
        ],
        "instant_meal": [
            "Homemade poha / upma instead of instant noodles",
            "Whole wheat pasta with vegetables",
            "Khichdi with dal and vegetables"
        ],
        "other": [
            "Prefer foods with shorter ingredient lists",
            "Choose products higher in fiber and lower in sugar",
            "Cook from basic ingredients when possible"
        ]
    }

    
    if processing_level == "Minimally processed":
        return [
            "This product already looks quite simple.",
            "You can still balance it with fruits, vegetables and enough water."
        ]

    category_key = category if category in alternatives else "other"
    return alternatives[category_key]


def calculate_health_score(nutrition):
    """
    Calculate a very rough health score out of 100.
    This is JUST for project demonstration. NOT for medical use.
    Uses: numpy array, basic math.
    nutrition = dict like:
    {
        'energy_kcal': ...,
        'sugar_g': ...,
        'sat_fat_g': ...,
        'sodium_mg': ...,
        'fiber_g': ...,
        'protein_g': ...
    }
    """
    
    values = np.array([
        nutrition["sugar_g"],
        nutrition["sat_fat_g"],
        nutrition["sodium_mg"],
        nutrition["fiber_g"],
        nutrition["protein_g"]
    ])

   
    weights = np.array([-1.5, -2.0, -0.01, 2.0, 1.0])

   
    raw_score = 100 + np.dot(weights, values)

    
    final_score = float(np.clip(raw_score, 0, 100))
    return final_score


def build_nutrition_dataframe(nutrition):
    """
    Create a pandas DataFrame from nutrition dict.
    Uses: pandas DataFrame
    """
    df = pd.DataFrame({
        "Nutrient": [
            "Energy (kcal)",
            "Sugar (g)",
            "Saturated fat (g)",
            "Sodium (mg)",
            "Fiber (g)",
            "Protein (g)"
        ],
        "Value per 100 g": [
            nutrition["energy_kcal"],
            nutrition["sugar_g"],
            nutrition["sat_fat_g"],
            nutrition["sodium_mg"],
            nutrition["fiber_g"],
            nutrition["protein_g"]
        ]
    })

    return df


def plot_nutrition_bar_chart(df):
    """
    Make a simple bar chart using matplotlib.
    Uses: matplotlib for data visualization.
    """
    fig, ax = plt.subplots()
    ax.bar(df["Nutrient"], df["Value per 100 g"])
    ax.set_xticklabels(df["Nutrient"], rotation=45, ha="right")
    ax.set_ylabel("Amount")
    ax.set_title("Nutrition per 100 g")
    plt.tight_layout()
    return fig



def main():
    st.title("Food Processing Level Checker 🥫")
    st.write("**Student Project – Python + Food Technology**")
    st.write("Upload a package photo (optional), then type ingredients and nutrition values.")

    
    uploaded_image = st.file_uploader(
        "Upload food package photo (optional, just for display)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded package image", use_column_width=True)

    st.subheader("Step 1: Enter Ingredients")
    st.write("Copy the ingredients from the packet and paste/type here.")

    ingredients_text = st.text_area(
        "Ingredients (separate by comma or new line)",
        height=120,
        placeholder="Example: Wheat flour, sugar, vegetable oil, emulsifier (INS 322), salt"
    )

    st.subheader("Step 2: Enter Nutrition Values per 100 g (from label)")
    col1, col2, col3 = st.columns(3)

    with col1:
        energy_kcal = st.number_input("Energy (kcal)", min_value=0.0, step=1.0, value=0.0)
        sugar_g = st.number_input("Sugar (g)", min_value=0.0, step=0.5, value=0.0)

    with col2:
        sat_fat_g = st.number_input("Saturated fat (g)", min_value=0.0, step=0.5, value=0.0)
        sodium_mg = st.number_input("Sodium (mg)", min_value=0.0, step=10.0, value=0.0)

    with col3:
        fiber_g = st.number_input("Fiber (g)", min_value=0.0, step=0.5, value=0.0)
        protein_g = st.number_input("Protein (g)", min_value=0.0, step=0.5, value=0.0)

    if st.button("Analyze Product"):
        if not ingredients_text.strip():
            st.warning("Please enter the ingredients first.")
        else:
            
            nutrition = {
                "energy_kcal": energy_kcal,
                "sugar_g": sugar_g,
                "sat_fat_g": sat_fat_g,
                "sodium_mg": sodium_mg,
                "fiber_g": fiber_g,
                "protein_g": protein_g
            }

            
            level, reason = classify_processing_level(ingredients_text)
            st.subheader("Processing Level Result")
            st.write(f"**Level:** {level}")
            st.write(f"**Why:** {reason}")


            health_score = calculate_health_score(nutrition)
            st.subheader("Simple Health Score (0 = less healthy, 100 = more healthy)")
            st.write(f"**Score:** {health_score:.1f} / 100")

        
            df = build_nutrition_dataframe(nutrition)
            st.subheader("Nutrition Table (per 100 g)")
            st.dataframe(df)

        
            st.subheader("Nutrition Visualisation")
            fig = plot_nutrition_bar_chart(df)
            st.pyplot(fig)

    
            category = guess_product_category(ingredients_text)
            suggestions = suggest_healthier_alternatives(category, level)

            st.subheader("Suggested Healthier Alternatives")
            for item in suggestions:
                st.write(f"- {item}")

            st.info("Note: This is a **simple educational project**. "
                    "The calculations are rough and not for medical decisions.")


if __name__ == "__main__":
    main()
