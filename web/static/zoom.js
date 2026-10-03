/* Reusable zoom modal for img.zoomable.
 *
 * Open:  click a .zoomable image (attachZoom() binds them, idempotent).
 * Zoom:  mouse wheel around the cursor (0.5x–5x), + / − / Reset buttons,
 *        double-click toggles 1x <-> 2.5x.
 * Pan:   drag while zoomed (grab / grabbing cursors).
 * Close: X button, Escape, or click on the backdrop (not the image).
 * The page behind stays put: body scroll is locked with a scrollbar-width
 * compensation so nothing reflows or resizes.
 */
(function () {
  "use strict";

  var MIN_SCALE = 0.5, MAX_SCALE = 5, STEP = 1.25, DBL_SCALE = 2.5;

  var modal = null, stage = null, imgEl = null, levelEl = null;
  var scale = 1, tx = 0, ty = 0;
  var dragging = false, lastX = 0, lastY = 0;

  function build() {
    modal = document.createElement("div");
    modal.className = "zoom-modal";
    modal.hidden = true;
    modal.innerHTML =
      '<div class="zoom-stage">' +
        '<img class="zoom-img" alt="">' +
      "</div>" +
      '<button type="button" class="zoom-x" title="Close (Esc)" aria-label="Close">\u00d7</button>' +
      '<div class="zoom-tools">' +
        '<button type="button" data-act="out" aria-label="Zoom out">\u2212</button>' +
        '<span class="zoom-level">100%</span>' +
        '<button type="button" data-act="in" aria-label="Zoom in">+</button>' +
        '<button type="button" data-act="reset" title="Reset zoom">Reset</button>' +
      "</div>";
    document.body.appendChild(modal);

    stage = modal.querySelector(".zoom-stage");
    imgEl = modal.querySelector(".zoom-img");
    levelEl = modal.querySelector(".zoom-level");

    modal.addEventListener("click", function (e) {
      if (e.target === modal || e.target === stage) close();
    });
    modal.querySelector(".zoom-x").addEventListener("click", close);
    modal.querySelector('[data-act="in"]').addEventListener("click", function () {
      zoomAt(viewportX(), viewportY(), scale * STEP);
    });
    modal.querySelector('[data-act="out"]').addEventListener("click", function () {
      zoomAt(viewportX(), viewportY(), scale / STEP);
    });
    modal.querySelector('[data-act="reset"]').addEventListener("click", reset);

    stage.addEventListener("wheel", function (e) {
      e.preventDefault();
      zoomAt(e.clientX, e.clientY, scale * Math.exp(-e.deltaY * 0.0015));
    }, {passive: false});

    imgEl.addEventListener("dblclick", function (e) {
      e.preventDefault();
      if (scale > 1.01) reset();
      else zoomAt(e.clientX, e.clientY, DBL_SCALE);
    });

    imgEl.addEventListener("pointerdown", function (e) {
      if (scale <= 1.01 || e.button !== 0) return;
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      stage.classList.add("grabbing");
      e.preventDefault();
    });
    window.addEventListener("pointermove", function (e) {
      if (!dragging) return;
      tx += e.clientX - lastX;
      ty += e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      apply();
    });
    window.addEventListener("pointerup", function () {
      if (!dragging) return;
      dragging = false;
      stage.classList.remove("grabbing");
      apply();
    });
    document.addEventListener("keydown", function (e) {
      if (!modal.hidden && e.key === "Escape") close();
    });
  }

  function viewportX() { return window.innerWidth / 2; }
  function viewportY() { return window.innerHeight / 2; }

  function apply() {
    imgEl.style.transform =
      "translate(" + tx + "px," + ty + "px) scale(" + scale + ")";
    levelEl.textContent = Math.round(scale * 100) + "%";
    stage.classList.toggle("grab", scale > 1.01 && !dragging);
  }

  /* Zoom so the image point under (cx, cy) stays under (cx, cy).
     With transform-origin at the image center and transform
     translate(tx,ty) scale(s), the image center sits at C0+(tx,ty); the
     correction for a scale factor k is therefore v*(1-k) where v is the
     cursor offset from the current (translated) center. */
  function zoomAt(cx, cy, next) {
    next = Math.max(MIN_SCALE, Math.min(MAX_SCALE, next));
    if (next === scale) return;
    var r = imgEl.getBoundingClientRect();
    var c0x = r.left + r.width / 2 - tx;
    var c0y = r.top + r.height / 2 - ty;
    var vx = cx - c0x - tx;
    var vy = cy - c0y - ty;
    var k = next / scale;
    tx += vx * (1 - k);
    ty += vy * (1 - k);
    scale = next;
    apply();
  }

  function reset() {
    scale = 1;
    tx = 0;
    ty = 0;
    if (imgEl) apply();
  }

  function open(src, alt) {
    if (!modal) build();
    imgEl.src = src;
    imgEl.alt = alt || "zoomed image";
    reset();
    // Compensate for the disappearing scrollbar so the page doesn't reflow.
    var sbw = window.innerWidth - document.documentElement.clientWidth;
    document.body.style.paddingRight = sbw > 0 ? sbw + "px" : "";
    document.body.classList.add("zoom-open");
    modal.hidden = false;
  }

  function close() {
    if (!modal || modal.hidden) return;
    modal.hidden = true;
    imgEl.removeAttribute("src");
    reset();
    document.body.classList.remove("zoom-open");
    document.body.style.paddingRight = "";
  }

  /* Bind click-to-open on every not-yet-bound img.zoomable under `root`. */
  function attachZoom(root) {
    var scope = root || document;
    var imgs = scope.querySelectorAll("img.zoomable");
    for (var i = 0; i < imgs.length; i++) {
      var im = imgs[i];
      if (im.dataset.zoomBound) continue;
      im.dataset.zoomBound = "1";
      im.addEventListener("click", function () { open(this.src, this.alt); });
    }
  }

  window.attachZoom = attachZoom;
})();
