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

    /* ========= DOM ELEMENTS ========= */
    const dayContextSelect = document.getElementById("dayContext");
    const timeContextSelect = document.getElementById("timeContext");
    const recommendBtn = document.querySelector(".recommend-btn");

    /* ========= RESTORE SESSION ========= */
    const currentUser = App.getUserContext();
    const userIdInput = document.getElementById("userIdInput");

    if (currentUser && currentUser.user_id) {
        console.log("[UI] Restoring session for User:", currentUser.user_id);

        // Auto-fill Input
        if (userIdInput) userIdInput.value = currentUser.user_id;

        // Auto-fill Context
        if (dayContextSelect) dayContextSelect.value = currentUser.is_weekend ? "weekend" : "weekday";
        if (timeContextSelect) timeContextSelect.value = currentUser.time_bucket || "morning";

        // Auto-fetch recommendations
        App.requestRecommendation();
    }

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
    // Vars declared at top

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

    /* ========= EVALUATION PANEL ========= */
    const evalBtn = document.getElementById("runEvalBtn");
    const evalUserCount = document.getElementById("evalUserCount");
    const evalResults = document.getElementById("evalResults");

    if (evalBtn) {
        evalBtn.addEventListener("click", async () => {
            const maxUsers = parseInt(evalUserCount?.value || "20");

            // Show loading
            evalBtn.disabled = true;
            evalBtn.textContent = "Đang chạy...";
            evalResults.innerHTML = `<p class="eval-loading">⏳ Đang đánh giá ${maxUsers} users... (có thể mất 1-2 phút)</p>`;

            try {
                const response = await Api.evaluate({ max_users: maxUsers, top_k: 10 });

                if (response.status === "success") {
                    const m = response.metrics;
                    const duration = response.duration_seconds || 0;
                    evalResults.innerHTML = `
                        <div class="eval-metric">
                            <span class="metric-label">Precision@10:</span>
                            <span class="metric-value">${(m["Precision@10"] * 100).toFixed(2)}%</span>
                        </div>
                        <div class="eval-metric">
                            <span class="metric-label">Recall@10:</span>
                            <span class="metric-value">${(m["Recall@10"] * 100).toFixed(2)}%</span>
                        </div>
                        <div class="eval-metric">
                            <span class="metric-label">HitRate@10:</span>
                            <span class="metric-value">${(m["HitRate@10"] * 100).toFixed(2)}%</span>
                        </div>
                        <div class="eval-metric">
                            <span class="metric-label">Rule Coverage:</span>
                            <span class="metric-value">${(m["RuleUserCoverage"] * 100).toFixed(1)}%</span>
                        </div>
                        <div class="eval-metric">
                            <span class="metric-label">Rule Item Share:</span>
                            <span class="metric-value">${(m["RuleItemShare"] * 100).toFixed(1)}%</span>
                        </div>
                        <p class="eval-note">✅ ${m["num_users_evaluated"]} users trong ${duration}s</p>
                    `;
                } else {
                    evalResults.innerHTML = `<p class="eval-error">❌ ${response.message}</p>`;
                }
            } catch (error) {
                console.error("[Eval] Error:", error);
                evalResults.innerHTML = `<p class="eval-error">❌ Lỗi: ${error.message}</p>`;
            } finally {
                evalBtn.disabled = false;
                evalBtn.textContent = "Chạy Evaluation";
            }
        });
    }

});
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

/* ========= UI RENDER FUNCTIONS ========= */
window.UI = window.UI || {};

/**
 * Render recommendations từ API response
 * @param {Object} response - { recommended_products, user, metadata }
 */
UI.renderRecommendations = function (response) {
    console.log("[UI] Rendering recommendations:", response);

    const grid = document.getElementById("productGrid");
    if (!grid) {
        console.error("[UI] productGrid not found");
        return;
    }

    // Clear existing products
    grid.innerHTML = "";

    const { recommended_products, user, metadata } = response;

    // ===== Update User Info Sidebar =====
    if (user) {
        const userIdEl = document.getElementById("uiUserId");
        const clusterEl = document.getElementById("uiUserCluster");
        const typeEl = document.getElementById("uiUserType");

        if (userIdEl) userIdEl.textContent = user.user_id || "---";

        if (user.cluster_info) {
            if (clusterEl) clusterEl.textContent = user.cluster_info.behavior_text || `Cụm ${user.cluster_info.behavior_cluster}`;
            if (typeEl) typeEl.textContent = user.cluster_info.lifecycle_text || user.cluster_info.lifecycle_stage || "---";
        }

        // ===== Render order history =====
        const orderList = document.getElementById("orderHistory");
        if (orderList && user.recent_purchases && user.recent_purchases.length > 0) {
            orderList.innerHTML = user.recent_purchases.slice(0, 5).map(item =>
                `<div class="order-item" title="ID: ${item.product_id}">${item.product_name}</div>`
            ).join("");
        }
    }

    // ===== Render Product Cards =====
    if (!recommended_products || recommended_products.length === 0) {
        grid.innerHTML = `<div class="no-results">Không có gợi ý cho user này</div>`;
        return;
    }

    recommended_products.forEach((item, index) => {
        const card = document.createElement("div");
        card.className = "product-card";
        card.dataset.id = item.item_id || item.product_id;
        card.dataset.name = item.product_name || `Sản phẩm #${item.item_id}`;
        card.dataset.price = item.price || 100000;

        // Calculate score percentage
        const scorePercent = item.score ? (item.score * 100).toFixed(1) : "N/A";
        const sources = item.source ? item.source.join(", ") : "N/A";
        const contextLevel = item.context_level ? item.context_level.join(", ") : "";
        const productName = item.product_name || `Sản phẩm #${item.item_id}`;
        const price = item.price || 100000;

        let levelHtml = "";
        if (contextLevel) {
            levelHtml = `<span style="background:#e0f2fe; color:#0369a1; padding:2px 6px; border-radius:4px; font-size:10px; margin-left:6px;">${contextLevel}</span>`;
        }

        card.innerHTML = `
            <p class="name" style="font-weight:bold; margin-bottom:8px;">${productName} ${levelHtml}</p>
            <p class="score" style="font-size:12px; color:#555;">Điểm: ${scorePercent}%</p>
            <p class="source" style="font-size:11px; color:#888;">${sources}</p>
            <p class="price" style="font-weight:bold; margin-top:8px;">${price.toLocaleString('vi-VN')}đ</p>
            <button class="add-product">Thêm</button>
        `;

        // Add to cart handler
        const addBtn = card.querySelector(".add-product");
        addBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            App.addToCart({
                product_id: item.item_id || item.product_id,
                name: productName,
                price: price
            });
            showToast(`🛒 Đã thêm "${productName}"`);
        });

        grid.appendChild(card);
    });

    console.log(`[UI] Rendered ${recommended_products.length} products`);
};

/**
 * Show loading state
 */
UI.showLoading = function (isLoading) {
    const grid = document.getElementById("productGrid");
    if (!grid) return;

    if (isLoading) {
        grid.innerHTML = `
            <div class="loading-state" style="text-align: center; padding: 40px; grid-column: 1/-1;">
                <p>Đang tải gợi ý...</p>
            </div>
        `;
    }
};

/**
 * Show error message
 */
UI.showError = function (message) {
    const grid = document.getElementById("productGrid");
    if (!grid) return;

    grid.innerHTML = `
        <div class="error-state" style="text-align: center; padding: 40px; color: #dc2626; grid-column: 1/-1;">
            <p>⚠️ ${message}</p>
        </div>
    `;
};

