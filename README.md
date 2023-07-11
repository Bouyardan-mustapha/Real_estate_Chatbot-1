# Real_estate_Chatbot
This project was created for the AI Hackathon 01 at FPN, organized by the I2A club and sponsored by Orbicall. Our project was selected as one of the winning projects among many other interesting ones.

Ce document est un guide décrivant notre projet. La première section est consacrée à une
description des différents fichiers utilisés. La deuxième section explique le mode d'utilisation du
chatbot, tandis que la troisième partie offre une vue détaillée du fonctionnement du chatbot et
de ses différents cas d'utilisation.
# 1. Les fichiers :
 app.js : ce fichier définit une classe Chatbox qui gère l'interaction de l'utilisateur avec
une boîte de discussion, envoie les messages de l'utilisateur à un endpoint de prédiction
et affiche les réponses du bot dans la boîte de discussion.

 style.css : ce fichier contient les styles CSS nécessaires pour mettre en forme et styliser
l'interface de la boîte de discussion. Ces styles incluent les couleurs, les dégradés, les
ombres de boîte et les bordures arrondies pour créer un design attrayant et cohérent.

 base.html : ce fichier représente la structure de base de la page HTML pour inclure le
chatbot et le rendre fonctionnel avec le style CSS et le scripts JavaScript décris
précédemment.

 nltk_utils.py : ce fichier contient des fonctions utilitaires pour le traitement du langage
naturel en utilisant la bibliothèque NLTK (Natural Language Toolkit). Ces fonctions sont :
tokenize(sentence), stem(word), bag_of_words(tokenized_sentence, words). Ces
méthodes sont utiles pour le prétraitement des données textuelles dans les tâches de
traitement du langage naturel, telles que la classification de texte ou la génération de
texte.

 modele.py : ce fichier contient la définition d'un modèle de réseau de neurones utilisant
PyTorch.

 train.py : ce code effectue le prétraitement des données, définit et entraîne un modèle
de réseau de neurones, puis sauvegarde le modèle entraîné sous forme d’un fichier
« data.pth » pour une utilisation ultérieure dans le chatbot.

 chat.py : code charge les données, définit les fonctions nécessaires pour traiter les
messages et générer des réponses, et exécute une boucle de chat qui attend les entrées
de l'utilisateur et affiche les réponses du chatbot.

 app.py : ce code définit les routes pour l'application Flask, gère les requêtes GET pour lapage d'accueil et les requêtes POST pour obtenir la réponse du chatbot. L'application est exécutée lorsqu'elle est exécutée directement.

 file.json : ce fichier JSON fourni est utilisé pour entraîner un modèle de chatbot. Il
contient plusieurs intentions (intents) classifiées avec des balises (tags), des modèles de
phrases (patterns) et des réponses adaptées (responses).

 prediction.ipynb : contient plusieurs algorithmes qui ont été testés pour trouver celui
offrant la meilleure précision (accuracy). Dans la première partie, le dataset a été utilisé
presque tel quel, et différents algorithmes ont été évalués. L'algorithme de Random
Forest a donné les meilleurs résultats par rapport aux autres, ce qui a conduit à la
création du fichier "random_forest_model.pkl". Ce fichier est utilisé ultérieurement
dans le chatbot pour prédire les prix.
Dans la deuxième partie, pour améliorer les performances du modèle,
RandomOverSampler a été utilisé. RandomOverSampler est une technique de
suréchantillonnage qui permet d'augmenter les données en créant des copies aléatoires
des échantillons sous-représentés dans le dataset. Cette augmentation des données a
été appliquée, puis le modèle Random Forest a été entraîné à nouveau avec ce nouvel
ensemble de données.
Les résultats de l'entraînement ont augmenté de 98,1% à 99% et les résultats du test
ont également augmenté de 87,6% à 99%.

 data.csv : ce fichier contient les données sur lesquels nous avons entrainé notre
modèle.

# 2. Mode d’utilisation :
Voici la procédure détaillée pour utiliser notre chatbot :
1. Ouvrez le terminal dans le répertoire où se trouve le projet.
2. Exécutez la commande "python train.py" (vous pouvez ignorer cette étape si vous n'avez
apporté aucune modification au fichier json et si le fichier data.pth existe déjà dans le dossier).
3. Si vous utilisez le modèle de prédiction "random_forest_model.pkl", vous pouvez passer à
l'étape suivante. Sinon, ouvrez le fichier "prediction.ipynb" et exécutez-le pour créer le fichier
"random_forest_model.pkl" (veillez à conserver le même nom). Il se peut que le modèle de
prédiction ne fonctionne pas dans l'exécution, donc il est recommandé de re-entraîner le
modèle et de remplacer le fichier existant par le nouveau.
4. Ensuite, exécutez la commande "python app.py" pour lancer l'application. Un messages'affichera dans la console, contenant l'adresse du serveur http://127.0.0.1:5000.
5. Copiez cette adresse dans votre navigateur, et le site web s'affichera.
6. Vous verrez une icône de chat en bas à droite de la page. Cliquez sur cette icône pour afficher
l'interface de chat.
7. Vous pouvez maintenant commencer à interagir avec le chatbot.
8. Si vous souhaitez quitter, vous pouvez appuyer sur CTRL + C dans la console.

# 3. Fonctionnement du chatbot :
Lorsque vous interagissez avec le chatbot, il répondra aux salutations et à la question "how are
you ?" Il peut également fournir son nom si vous le lui demandez. Puis, vous pouvez lui
demander de prédire le prix d'une maison. Il vous demandera ensuite de fournir les valeurs des
caractéristiques de la maison suivantes : 'bedrooms' (nombre de chambres), 'bathrooms'
(nombre de salles de bains), 'totallivingarea' (surface habitable totale), 'totallotarea' (surface
totale du terrain), 'floors' (nombre d'étages), 'waterfront' (proximité de l'eau, soit 0 soit 1),
'view' (vue, valeur entre 0 et 5), 'condition' (état général, valeur entre 1 et 5), 'grade' (niveau de
qualité, valeur entre 1 et 13), 'above' (surface du rez-de-chaussée), 'basement' (surface du
sous-sol), 'built' (année de construction, entre 1900 et 2021), 'renovated' (année de rénovation,
inférieure à 2022), 'zipcode' (code postal à 5 chiffres), 'latitude' (latitude de la maison),
'longitude' (longitude de la maison), 'livingarea' (surface habitable) et 'lotarea' (superficie du
terrain).

Si une condition n'est pas respectée, le chatbot vous demandera de saisir à nouveau la valeur
en précisant les conditions requises. Vous n'êtes pas obligé de donner les valeurs dans l'ordre
demandé par le chatbot ; vous pouvez fournir le tag et la valeur correspondante, et le chatbot
se chargera de les enregistrer au bon endroit. Si le chatbot demande une valeur pour un tag
spécifique, vous pouvez simplement fournir la valeur sans le tag, et le chatbot comprendra qu'il
s'agit de la valeur pour le tag demandé. Si vous fournissez uniquement le tag sans valeur, le
chatbot vous demandera la valeur correspondante.

Il est possible de mettre à jour une valeur dans la table en fournissant le tag et la nouvelle
valeur, et le chatbot se chargera de la modifier. Les valeurs contenant un point '.' ou un tiret '-'
(pour la négation) sont acceptées pour les coordonnées de latitude et longitude.
Une fois que le chatbot a reçu toutes les valeurs nécessaires, il vous retournera la prédiction du
prix. Vous pouvez également mettre à jour une valeur de la même manière décrite
précédemment, et le chatbot recalculera la prédiction avec la nouvelle valeur.
Si vous souhaitez recommencer depuis le début en saisissant de nouvelles valeurs, vous pouvez
taper "restart" (même si vous incluez le mot "restart" dans une phrase, cela fonctionnera). Dans
ce cas, le chatbot supprimera les valeurs précédemment fournies et recommencera la demandedes valeurs à zéro.
À la fin de la conversation, le chatbot est également capable de reconnaître lorsque l'utilisateur
lui dit "Goodbye" et il répondra de manière appropriée en vous souhaitant une bonne journée.
N'hésitez pas à utiliser cette formulation pour terminer la conversation de manière courtoise.
