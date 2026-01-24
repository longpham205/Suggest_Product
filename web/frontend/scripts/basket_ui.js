// scripts/basket_ui.js
document.addEventListener("DOMContentLoaded", () => {
    const cartContainer = document.getElementById("cartContainer");
    const summaryProduct = document.querySelector(".summary-left span");
    const summaryPrice = document.querySelector(".total-price");

    /* ======================
       Utils
    ====================== */
    function getCart() {
        return JSON.parse(localStorage.getItem("cart")) || [];
    }

    function setCart(cart) {
        localStorage.setItem("cart", JSON.stringify(cart));
    }

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

        summaryProduct.innerHTML =
            `<strong>Tổng sản phẩm:</strong> ${totalQty}`;

        summaryPrice.innerText =
            `Tổng tiền: ${formatPrice(totalMoney)}`;
    }

    /* ======================
       Render Cart
    ====================== */
    function renderCart() {
        const cart = getCart();
        cartContainer.innerHTML = "";

        if (cart.length === 0) {
            cartContainer.innerHTML = `<p class="empty-cart">Giỏ hàng trống</p>`;
            updateSummary([]);
            return;
        }

        cart.forEach((item, index) => {
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
               Events
            ====================== */
            row.querySelector(".plus").onclick = () => {
                cart[index].qty++;
                setCart(cart);
                renderCart();
            };

            row.querySelector(".minus").onclick = () => {
                if (cart[index].qty === 1) {
                    if (!confirm("Bạn có chắc chắn muốn xoá sản phẩm này không?")) {
                        return;
                    }
                    cart.splice(index, 1);
                } else {
                    cart[index].qty--;
                }

                setCart(cart);
                renderCart();
            };

            row.querySelector(".delete-btn").onclick = () => {
                if (confirm("Bạn có chắc chắn muốn xoá sản phẩm này không?")) {
                    cart.splice(index, 1);
                    setCart(cart);
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
    renderCart();
});
