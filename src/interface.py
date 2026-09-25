
def predict_purchase(model, session_data):
    """Predict whether a shopping session will purchase."""

    if session_data is None or session_data.empty:
        return {
            "error": "Please provide shopping session data."
        }

    try:
        prediction = model.predict(session_data)[0]
        probability = model.predict_proba(
            session_data
        )[0, 1]

        return {
            "prediction": bool(prediction),
            "purchase_probability": float(probability)
        }

    except (ValueError, KeyError) as error:
        return {
            "error": "Invalid shopping session data: "
                     + str(error)
        }
