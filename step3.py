import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from step2 import get_embeddings

# Define list of commercial keywords related to buying and selling
commercial_keywords = ["vente", "achat", "شراء", "بيع"]
commercial_keywords_fr = [
    "restaurant", "vente", "achat", "café", "fast food", "boucherie", "poissonnerie", "téléphone", 
    "bijouterie", "magasin", "boutique", "fournisseur", "détaillant", "grossiste", "revendeur", 
    "importateur", "exportateur", "franchise", "vendeur", "commerçant", "négoce", "dépôt vente", 
    "magasin alimentaire", "produits alimentaires", "produits de consommation", "liquidation", 
    "bazar", "marché", "vente de voitures", "agents commerciaux", "services de livraison", 
    "construction", "matériel de construction", "équipement industriel", "téléphonie mobile", "réparation",
    "location", "distribution", "importation", "commerce de détail", "distributeur", "commerçant de proximité", 
    "agricole", "produits agricoles", "alimentation générale", "commerçant ambulant", "transports", "kms", "kiosque"
]
commercial_keywords_ar = [
    "مطعم", "بيع", "شراء", "مقهى", "وجبات سريعة", "جزارة", "سمك", "هاتف", "مجوهرات", "متجر", "دكان", 
    "مورد", "تاجر تجزئة", "تاجر جملة", "بائع", "مستورد", "مُصدر", "امتياز", "بائع", "تاجر", "تجارة", 
    "بيع بالتجزئة", "منتجات غذائية", "منتجات استهلاكية", "تصفية", "بازار", "سوق", "بيع السيارات", 
    "وكلاء تجاريين", "خدمات التوصيل", "إنشاء", "معدات البناء", "تجهيزات صناعية", "هواتف محمولة", 
    "صيانة", "إيجار", "توزيع", "استيراد", "تجارة التجزئة", "موزع", "تاجر محلي", "زراعي", "منتجات زراعية", 
    "تجارة عامة", "تاجر متجول", "نقل"
]
intermediary_keywords_fr = [
    "e-commerce", "plateforme de vente", "e-shop", "site marchand", "marketplace", "dropshipping", 
    "affiliation", "vente en ligne", "webshop", "commerce virtuel", "commerce digital", "fournisseur e-commerce", 
    "commerce électronique", "start-up", "entrepreneur digital", "services de paiement", "agent commercial", 
    "courtier", "publicité en ligne", "consultant", "services en ligne", "freelance", "plateforme de freelancing", 
    "commercialisation", "digital marketing", "plateforme B2B", "réseau de distribution"
]
intermediary_keywords_ar = [
    "تجارة إلكترونية", "منصة بيع", "متجر إلكتروني", "موقع تجاري", "سوق إلكتروني", "دروبشيبينغ", 
    "تسويق بالعمولة", "متجر على الإنترنت", "تجارة افتراضية", "تجارة رقمية", "مورد تجارة إلكترونية", 
    "شركة ناشئة", "ريادي رقمي", "خدمات الدفع", "وكيل تجاري", "سمسار", "إعلانات عبر الإنترنت", 
    "مستشار", "خدمات عبر الإنترنت", "مستقل", "منصة عمل مستقل", "تسويق", "تسويق رقمي", "إدارة الحملات", 
    "منصة إعلانات", "تسويق بالعمولة", "وسيط", "وكيل مبيعات", "حلول رقمية", "تخزين سحابي", "دعم العملاء", 
    "إعلانات رقمية", "وسائل التواصل الاجتماعي", "منصة بي تو بي", "شبكة توزيع"
]

# Function to check if any commercial keywords are in the activity or description
def is_commercial(activity, description):
    for keyword in commercial_keywords:
        if keyword in activity or keyword in description:
            return True
    return False

# Function to process the cleaned embeddings and filter out commercial activities
def filter_commercial_activities(cleaned_df):
    cleaned_df['is_commercial'] = cleaned_df.apply(
        lambda row: is_commercial(str(row['activity']), str(row['description'])), axis=1
    )

    # Exclude commercial activities by filtering out rows where 'is_commercial' is True
    non_commercial_cleaned = cleaned_df[cleaned_df['is_commercial'] == False]

    return non_commercial_cleaned

def classify_activity(row, commercial_keywords_df, intermediary_keywords_df):
    # Get the activity's embedding
    activity_embedding = row['embeddings']
    
    # Calculate cosine similarity between activity embedding and commercial keywords embeddings
    commercial_similarity = cosine_similarity([activity_embedding], commercial_keywords_df['embedding'].tolist())
    intermediary_similarity = cosine_similarity([activity_embedding], intermediary_keywords_df['embedding'].tolist())
    
    # Find the max similarity score for commercial and intermediary
    max_commercial_sim = np.max(commercial_similarity)
    max_intermediary_sim = np.max(intermediary_similarity)
    
    # If the max similarity score is high for commercial keywords, mark it as commercial
    if max_commercial_sim > max_intermediary_sim:
        return True  # This is a commercial activity
    else:
        return False  # This is either intermediary or does not match any commercial keyword

# Function to filter out commercial activities from the DataFrame
def filter_commercial_activities_advanced(cleaned_df):
    # Create DataFrame for Commercial French
    df_commercial_fr = pd.DataFrame({'keyword': commercial_keywords_fr})
    df_commercial_fr['embedding'] = df_commercial_fr['keyword'].apply(get_embeddings)
    
    # Create DataFrame for Commercial Arabic
    df_commercial_ar = pd.DataFrame({'keyword': commercial_keywords_ar})
    df_commercial_ar['embedding'] = df_commercial_ar['keyword'].apply(get_embeddings)
    
    # Create DataFrame for Intermediary French
    df_intermediary_fr = pd.DataFrame({'keyword': intermediary_keywords_fr})
    df_intermediary_fr['embedding'] = df_intermediary_fr['keyword'].apply(get_embeddings)
    
    # Create DataFrame for Intermediary Arabic
    df_intermediary_ar = pd.DataFrame({'keyword': intermediary_keywords_ar})
    df_intermediary_ar['embedding'] = df_intermediary_ar['keyword'].apply(get_embeddings)
    
    # Apply the classification function to the cleaned DataFrame
    cleaned_df['is_commercial'] = cleaned_df.apply(
        classify_activity, axis=1, commercial_keywords_df=df_commercial_fr, intermediary_keywords_df=df_intermediary_fr
    )

    # Filter the dataframe to keep only non-commercial activities
    filtered_df = cleaned_df[cleaned_df['is_commercial'] == False]
    
    return filtered_df
