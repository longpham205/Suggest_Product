// scripts/basket_ui.js
document.addEventListener("DOMContentLoaded", () => {

    // Check if App is ready
    if (!window.App) {
        console.error("[Basket UI] App not loaded");
        return;
    }

    const cartContainer = document.getElementById("cartContainer");
    const summaryProduct = document.querySelector(".summary-left span");
    const summaryPrice = document.querySelector(".total-price");
    const totalQtyEl = document.getElementById("totalQty");
    const totalPriceEl = document.getElementById("totalPrice");

    /* ======================
       Utils
    ====================== */
    function formatPrice(number) {
        return number.toLocaleString("vi-VN") + "đ";
    }

    /* ======================
       Summary
    ====================== */
    function updateSummary(cart) {
        let totalQty = 0;
        let totalMoney = 0;

        cart.forEach(item => {
            totalQty += item.qty;
            totalMoney += item.qty * item.price;
        });

        if (totalQtyEl) totalQtyEl.innerText = totalQty;
        if (totalPriceEl) totalPriceEl.innerText = formatPrice(totalMoney);
    }

    /* ======================
       Render Cart
    ====================== */
    function renderCart() {
        // Use App.getCart() to get the correct user's cart
        const cart = App.getCart();
        cartContainer.innerHTML = "";

        if (cart.length === 0) {
            const user = App.getUserContext();
            const userId = user.user_id;
            const contextRaw = localStorage.getItem("user_context");
            const cartKey = userId ? `cart_user_${userId}` : "N/A";
            const cartRaw = userId ? localStorage.getItem(cartKey) : "N/A";

            const debugInfo = `
                <div style="font-size:12px; color:#666; margin-top:10px; text-align:left; background:#f5f5f5; padding:10px; border-radius:4px;">
                    <strong>Debug Info:</strong><br/>
                    User ID in App: ${userId || "null"}<br/>
                    LocalStorage [user_context]: ${contextRaw || "null"}<br/>
                    Expected Cart Key: ${cartKey}<br/>
                    LocalStorage [${cartKey}]: ${cartRaw ? "Found data" : "null/empty"}
                </div>
            `;

            cartContainer.innerHTML = `<p class="empty-cart" style="text-align:center; padding:20px;">
                Giỏ hàng trống<br/>
                <a href="index.html">Quay lại mua sắm</a>
            </p>${debugInfo}`;
            updateSummary([]);
            return;
        }

        cart.forEach((item) => {
            const row = document.createElement("div");
            row.className = "cart-row";

            row.innerHTML = `
                <div class="product-name">${item.name}</div>

                <div class="unit-price">
                    ${formatPrice(item.price)}
                </div>

                <div class="quantity-control">
                    <button class="qty-btn minus">-</button>
                    <span class="qty">${item.qty}</span>
                    <button class="qty-btn plus">+</button>
                </div>

                <div class="row-total">
                    ${formatPrice(item.price * item.qty)}
                </div>

                <button class="delete-btn">Xoá</button>
            `;

            /* ======================
               Events (Use App methods)
            ====================== */
            row.querySelector(".plus").onclick = () => {
                App.updateCartQty(item.product_id, 1);
                renderCart();
            };

            row.querySelector(".minus").onclick = () => {
                if (item.qty === 1) {
                    if (!confirm("Bạn có chắc chắn muốn xoá sản phẩm này không?")) return;
                }
                App.updateCartQty(item.product_id, -1);
                renderCart();
            };

            row.querySelector(".delete-btn").onclick = () => {
                if (confirm("Bạn có chắc chắn muốn xoá sản phẩm này không?")) {
                    App.removeFromCart(item.product_id);
                    renderCart();
                }
            };

            cartContainer.appendChild(row);
        });

        updateSummary(cart);
    }

    /* ======================
       Init
    ====================== */
    // Wait for App to initialize data from localStorage
    setTimeout(() => {
        renderCart();
    }, 100);
});
