// scripts/main.js

window.App = (function () {
    /* ========= GLOBAL STATE ========= */

    let state = {
        user: {
            user_id: null,
            time_bucket: "morning",
            is_weekend: false
        },
        cart: [],
        recommendations: [],
        cartBoost: null
    };

    const CART_KEY = "cart";

    /* ========= INTERNAL EVENTS ========= */

    function notifyCartChanged() {
        const totalQty = state.cart.reduce((sum, i) => sum + i.qty, 0);

        document.dispatchEvent(
            new CustomEvent("cart:changed", {
                detail: {
                    totalQty,
                    cart: [...state.cart]
                }
            })
        );
    }

    /* ========= INIT ========= */

    function init() {
        loadCartFromStorage();
        notifyCartChanged(); // 🔥 update badge ngay khi load
        console.log("[App] Initialized", state);
    }

    /* ========= USER CONTEXT ========= */

    function setUserContext({ user_id, time_bucket, is_weekend }) {
        state.user.user_id = Number(user_id);
        state.user.time_bucket = time_bucket;
        state.user.is_weekend = Boolean(is_weekend);

        console.log("[App] User context set:", state.user);
    }

    function getUserContext() {
        return state.user;
    }

    /* ========= CART STATE ========= */

    function loadCartFromStorage() {
        state.cart = JSON.parse(localStorage.getItem(CART_KEY)) || [];
    }

    function saveCartToStorage() {
        localStorage.setItem(CART_KEY, JSON.stringify(state.cart));
    }

    function getCart() {
        return [...state.cart];
    }

    function addToCart(product) {
        /**
         * product = {
         *   product_id,
         *   name,
         *   price
         * }
         */

        const existing = state.cart.find(
            item => item.product_id === product.product_id
        );

        if (existing) {
            existing.qty += 1;
        } else {
            state.cart.push({
                product_id: product.product_id,
                name: product.name,
                price: product.price,
                qty: 1
            });
        }

        saveCartToStorage();
        notifyCartChanged(); // 🔥 badge update

        console.log("[App] Added to cart:", product);

        // trigger cart boost
        requestCartBoost(product.product_id);
    }

    function updateCartQty(product_id, delta) {
        const item = state.cart.find(i => i.product_id === product_id);
        if (!item) return;

        item.qty += delta;

        if (item.qty <= 0) {
            state.cart = state.cart.filter(i => i.product_id !== product_id);
        }

        saveCartToStorage();
        notifyCartChanged(); // 🔥 badge update
    }

    function removeFromCart(product_id) {
        state.cart = state.cart.filter(i => i.product_id !== product_id);
        saveCartToStorage();
        notifyCartChanged(); // 🔥 badge update
    }

    /* ========= RECOMMENDATION ========= */

    async function requestRecommendation() {
        if (!state.user.user_id) {
            console.warn("[App] user_id not set");
            return;
        }

        const payload = {
            user_id: state.user.user_id,
            context: {
                time_bucket: state.user.time_bucket,
                is_weekend: state.user.is_weekend
            },
            cart_items: state.cart.map(i => i.product_id)
        };

        try {
            const res = await Api.recommend(payload);
            state.recommendations = res.recommended_products || [];

            console.log("[App] Recommendations received", res);

            if (window.UI && UI.renderRecommendations) {
                UI.renderRecommendations(res);
            }
        } catch (err) {
            console.error("[App] Recommend error", err);
        }
    }

    /* ========= CART BOOST ========= */

    async function requestCartBoost(added_product_id) {
        if (!state.user.user_id) return;

        const payload = {
            user_id: state.user.user_id,
            added_product_id,
            context: {
                time_bucket: state.user.time_bucket,
                is_weekend: state.user.is_weekend
            },
            cart_items: state.cart.map(i => i.product_id)
        };

        try {
            const res = await Api.cartBoost(payload);
            state.cartBoost = res;

            console.log("[App] Cart boost received", res);

            if (window.UI && UI.renderCartBoost) {
                UI.renderCartBoost(res);
            }
        } catch (err) {
            console.error("[App] Cart boost error", err);
        }
    }

    /* ========= PUBLIC API ========= */

    return {
        init,

        // user
        setUserContext,
        getUserContext,

        // cart
        getCart,
        addToCart,
        updateCartQty,
        removeFromCart,

        // recommend
        requestRecommendation
    };
})();

/* ========= AUTO INIT ========= */

document.addEventListener("DOMContentLoaded", () => {
    App.init();
});
