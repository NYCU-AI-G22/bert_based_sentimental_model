# bert_based_sentimental_model


## Download the Saved Model

Download the model from the following link and place it with the code:

[Saved Model](https://drive.google.com/file/d/10U6s4XJGkLMPC8M7lKmq1Q-s7o7EKkrB/view?usp=sharing)

## Getting Started

**Run the Flask application**:

    ```
    python run.py
    ```

    After running the above command, you should see the following output indicating that the server is running:

    ```
    * Running on http://{YOUR LOCALHOST}:3000/
    ```

## Using the API

You can use `curl` to interact with the API. Here is an example of how to use the `text_generation` endpoint:

```sh
curl -X POST http://{YOUR LOCALHOST}:3000/text_generation -H "Content-Type: application/json" -d "{\"input_text\": \"I was sad that my mom abandon me.\"}"
```

This command sends a POST request to the `text_generation` endpoint with a JSON payload containing the input text.
