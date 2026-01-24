// scripts/api.js

window.Api = (function () {

    /* ========= CONFIG ========= */

    // 👉 chỉnh đúng URL backend của bạn ở đây
    const BASE_URL = "http://localhost:8000";

    const DEFAULT_HEADERS = {
        "Content-Type": "application/json"
    };

    /* ========= CORE FETCH ========= */

    async function request(path, method = "POST", body = null) {
        const options = {
            method,
            headers: DEFAULT_HEADERS
        };

        if (body) {
            options.body = JSON.stringify(body);
        }

        const response = await fetch(`${BASE_URL}${path}`, options);

        if (!response.ok) {
            const text = await response.text();
            throw new Error(
                `[API ERROR] ${method} ${path} (${response.status}): ${text}`
            );
        }

        return response.json();
    }

    /* ========= RECOMMEND ========= */

    /**
     * payload = {
     *   user_id,
     *   context: { time_bucket, is_weekend },
     *   cart_items: [product_id, ...]
     * }
     */
    function recommend(payload) {
        console.log("[API] /recommend", payload);
        return request("/recommend", "POST", payload);
    }

    /* ========= CART BOOST ========= */

    /**
     * payload = {
     *   user_id,
     *   added_product_id,
     *   context: { time_bucket, is_weekend },
     *   cart_items: [product_id, ...]
     * }
     */
    function cartBoost(payload) {
        console.log("[API] /cart/boost", payload);
        return request("/cart/boost", "POST", payload);
    }

    /* ========= HEALTH (OPTIONAL) ========= */

    function health() {
        return request("/health", "GET");
    }

    /* ========= PUBLIC API ========= */

    return {
        recommend,
        cartBoost,
        health
    };

})();
