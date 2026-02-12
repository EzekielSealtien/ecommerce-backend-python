from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

text = """
Notre site e-commerce permet de commander facilement des produits neufs et 
authentiques en les ajoutant au panier puis en suivant les étapes de paiement,
avec ou sans création de compte, et une confirmation de commande est envoyée par
courriel après validation. Les produits peuvent parfois être en rupture de stock, 
mais les clients ont la possibilité de s’inscrire pour être informés de leur 
disponibilité, et des guides détaillés aident à choisir la bonne taille ou le bon 
modèle. Les paiements sont sécurisés par des protocoles SSL et peuvent être effectués
par carte de crédit, PayPal ou d’autres moyens selon la région, bien qu’un refus de 
paiement puisse survenir en cas d’informations incorrectes ou de restrictions 
bancaires. Les commandes sont généralement livrées sous 3 à 7 jours ouvrables, avec 
un numéro de suivi envoyé dès l’expédition, et des options de livraison 
internationale sont proposées selon la destination. Les clients disposent de 14 
jours après réception pour retourner un produit non utilisé dans son emballage 
d’origine, et les remboursements sont traités dans un délai de 5 à 10 jours 
ouvrables après réception du retour. La création d’un compte, bien que facultative, 
permet de suivre les commandes, de gérer les informations personnelles et de 
récupérer un mot de passe en cas d’oubli. Le service client est accessible par 
formulaire, courriel ou chatbot du lundi au vendredi de 9h à 18h, et toutes les 
données personnelles sont protégées conformément aux lois en vigueur sur la confidentialité."""

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
)

embedding = response.data[0].embedding

# Vérification importante
print("Dimension de l'embedding :", len(embedding))

# Affichage prêt à coller dans Pinecone (Dense values)
dense_values = ", ".join(str(x) for x in embedding)
print(dense_values)

with open("embedding.txt", "w") as f:
    f.write(dense_values)
