from flask import Flask, request, jsonify

app = Flask(__name__)


def serialise_entity_pairs(entity_a, entity_b, weight):
    try:
        entities = [entity_a, entity_b]
        entity_pairs = []
        for entity in entities:
            entity_series = ""
            for key, val in entity.items():
                if key != "_id":
                    key = key.replace("\n", "")
                    val = val.replace("\n", "")

                    key = key.replace("\t", " ")
                    val = val.replace("\t", " ")

                    entity_series += f"COL {key} VAL {val} "
            entity_series += f"COL weight VAL {weight}"
            entity_series = entity_series.strip()
            entity_pairs.append(entity_series)
        entity_serialized = '\t'.join(entity_pairs)

        return entity_serialized
    except Exception as args:
        raise f"Error while serialization of entity pairs. ERROR: {args}"


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        entity_pairs_series = serialise_entity_pairs(data['entity_a'], data['entity_b'], data['weight'])

        return jsonify({
            'class': 1,
            'confidence': 0.99
        })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
