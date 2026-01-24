// scripts/index_ui.js
// =======================================
// INDEX UI (UI ONLY - FULL & CLEAN)
// =======================================

document.addEventListener("DOMContentLoaded", () => {

    if (!window.App) {
        console.error("[UI] App is not loaded");
        return;
    }

    /* ========= CART BADGE ========= */
    const cartCountEl = document.getElementById("cartCount");

    function updateCartBadge() {
        if (!cartCountEl) return;

        const cart = App.getCart(); // dùng public API
        const totalQty = cart.reduce((sum, item) => sum + item.qty, 0);

        cartCountEl.innerText = totalQty;
        cartCountEl.style.display = totalQty > 0 ? "inline-block" : "none";

        // UX nhỏ xịn ✨
        cartCountEl.classList.add("pop");
        setTimeout(() => cartCountEl.classList.remove("pop"), 150);
    }

    // Sync badge khi load trang
    updateCartBadge();

    /* ========= ORDER ITEM UI ========= */
    document.querySelectorAll(".order-item").forEach(item => {
        item.addEventListener("click", () => {
            item.classList.toggle("active");
        });
    });

    /* ========= ADD TO CART ========= */
    document.querySelectorAll(".product-card").forEach(card => {

        const addBtn = card.querySelector(".add-product");
        if (!addBtn) return;

        // ✅ đọc dataset đúng theo HTML
        const productId = card.dataset.id;     // ví dụ: "A01"
        const name = card.dataset.name;
        const price = Number(card.dataset.price);

        if (!productId || !name || !price) {
            console.warn("[UI] Missing product info", card);
            return;
        }

        addBtn.addEventListener("click", (e) => {
            e.stopPropagation();

            App.addToCart({
                product_id: productId,
                name,
                price
            });

            updateCartBadge();
            showToast(`🛒 Đã thêm "${name}"`);
        });
    });

    /* ========= CONTEXT CONTROLS ========= */
    const userIdInput = document.getElementById("userIdInput");
    const dayContextSelect = document.getElementById("dayContext");
    const timeContextSelect = document.getElementById("timeContext");
    const recommendBtn = document.querySelector(".recommend-btn");

    if (recommendBtn) {
        recommendBtn.addEventListener("click", () => {
            const user_id = Number(userIdInput.value);

            if (!user_id) {
                alert("Vui lòng nhập User ID");
                return;
            }

            App.setUserContext({
                user_id,
                time_bucket: timeContextSelect.value,
                is_weekend: dayContextSelect.value === "weekend"
            });

            App.requestRecommendation();
        });
    }
});

/* ========= TOAST UI ========= */
function showToast(message) {
    const toast = document.createElement("div");
    toast.innerText = message;

    Object.assign(toast.style, {
        position: "fixed",
        bottom: "30px",
        right: "30px",
        background: "#16a34a",
        color: "white",
        padding: "12px 20px",
        borderRadius: "12px",
        boxShadow: "0 8px 20px rgba(0,0,0,0.3)",
        zIndex: "9999",
        fontSize: "14px"
    });

    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 1600);
}
