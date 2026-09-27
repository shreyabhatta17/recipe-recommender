"""
Holds the artefacts loaded once by api.apps.ApiConfig.ready() at server
startup. A plain module used as a namespace -- views/services import this
module and read its attributes, rather than passing a dozen objects around.

`loaded` is False until ready() successfully populates everything; views
check it and return 503 rather than crashing on a missing attribute.
"""
loaded = False

vectorizer = None
tfidf_matrix = None
valid_mask = None
coll_model = None

user_index = None
item_index = None
index_item = None
recipe_id_to_row = None
row_to_recipe_id = None

best_alpha = None
best_lambda = None
cold_start_threshold = None

cb_kwargs = None
coll_kwargs = None
