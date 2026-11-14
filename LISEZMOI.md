# Relais Groq (à déployer sur Render)

Petit serveur qui reçoit un texte depuis l'Article Admin, appelle Groq et renvoie la réponse.
La clé Groq reste sur le serveur, jamais dans le navigateur.

## Déploiement

1. Crée un **nouveau dépôt GitHub privé** (par exemple `groq-relay`) et mets-y `main.py` et `requirements.txt`
   (les tests `test_relay.py` sont facultatifs).
2. Sur Render : **New > Web Service**, choisis ce dépôt, puis :
   - Runtime : Python
   - Build command : `pip install -r requirements.txt`
   - Start command : `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Dans **Environment**, ajoute :

| Variable | Valeur |
|---|---|
| `GROQ_API_KEY` | une clé Groq créée pour cet usage (avec limite de dépense si possible) |
| `RELAY_TOKEN` | un long mot de passe aléatoire que tu inventes (30 caractères ou plus) |
| `ALLOWED_ORIGINS` | `https://article-admin.private.zooplus.net` |
| `GROQ_MODEL` | facultatif, par défaut `llama-3.3-70b-versatile` |

4. Une fois déployé, ouvre `https://<ton-service>.onrender.com/health` dans un onglet : tu dois voir `{"ok":true}`.

## Remarques

- Avec l'offre gratuite de Render, le serveur s'endort après un moment d'inactivité : le **premier appel de la journée peut prendre environ une minute**. L'outil le gèrera.
- Ne mets jamais `RELAY_TOKEN` ni `GROQ_API_KEY` dans le code ou le dépôt, seulement dans les variables d'environnement de Render.
- Le relais refuse tout appel sans le jeton et toute page qui n'est pas dans `ALLOWED_ORIGINS`.
- Il limite la taille des messages et le nombre de tokens, pour éviter qu'une erreur ne vide ton crédit Groq.
