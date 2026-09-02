(() => {
  const article = document.querySelector("#article");
  const sideToc = document.querySelector("#toc-side");
  const sidebar = document.querySelector("#sidebar");
  const menuToggle = document.querySelector("#menu-toggle");
  const backdrop = document.querySelector("#drawer-backdrop");
  const themeToggle = document.querySelector("#theme-toggle");
  const progress = document.querySelector("#reading-progress");
  const backToTop = document.querySelector("#back-to-top");
  const dialog = document.querySelector("#figure-dialog");
  const dialogImage = document.querySelector("#figure-dialog-image");
  const dialogClose = document.querySelector("#figure-dialog-close");

  const slugCounts = new Map();
  const slugify = (value) => {
    const base = value
      .toLowerCase()
      .normalize("NFKD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "") || "section";
    const count = slugCounts.get(base) || 0;
    slugCounts.set(base, count + 1);
    return count ? `${base}-${count + 1}` : base;
  };

  const headings = [...article.querySelectorAll("h1, h2, h3, h4")];
  headings.forEach((heading) => {
    if (!heading.id) heading.id = slugify(heading.textContent.trim());
  });

  const parts = headings.filter((heading) => heading.tagName === "H2");

  parts.forEach((part) => {
    const link = document.createElement("a");
    link.href = `#${part.id}`;
    link.textContent = part.textContent;
    link.dataset.target = part.id;
    sideToc.append(link);
  });

  article.querySelectorAll("table").forEach((table) => {
    const wrapper = document.createElement("div");
    wrapper.className = "table-wrap";
    table.parentNode.insertBefore(wrapper, table);
    wrapper.append(table);
  });

  article.querySelectorAll("pre").forEach((pre) => {
    const button = document.createElement("button");
    button.className = "copy-code";
    button.type = "button";
    button.textContent = "Copy";
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(pre.querySelector("code")?.innerText || pre.innerText);
        button.textContent = "Copied";
        setTimeout(() => { button.textContent = "Copy"; }, 1400);
      } catch {
        button.textContent = "Select text";
      }
    });
    pre.append(button);
  });

  article.querySelectorAll("a[href^='http']").forEach((link) => {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  });

  const closeDrawer = () => {
    sidebar.classList.remove("open");
    backdrop.classList.remove("visible");
    if (menuToggle) {
      menuToggle.setAttribute("aria-expanded", "false");
      menuToggle.classList.remove("active");
    }
  };

  if (menuToggle) {
    menuToggle.addEventListener("click", () => {
      const willOpen = !sidebar.classList.contains("open");
      sidebar.classList.toggle("open", willOpen);
      backdrop.classList.toggle("visible", willOpen);
      menuToggle.setAttribute("aria-expanded", String(willOpen));
      menuToggle.classList.toggle("active", willOpen);
    });
  }
  backdrop.addEventListener("click", closeDrawer);
  sideToc.addEventListener("click", closeDrawer);

  let savedTheme = null;
  try { savedTheme = localStorage.getItem("satquery-theme"); } catch { /* file:// privacy mode */ }
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const applyTheme = (theme) => {
    document.documentElement.dataset.theme = theme;
    const nextTheme = theme === "dark" ? "light" : "dark";
    const icon = theme === "dark"
      ? '<svg class="theme-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M21 12.8A8.5 8.5 0 1 1 11.2 3 6.7 6.7 0 0 0 21 12.8Z" /></svg>'
      : '<svg class="theme-icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4" fill="#ffffff" /><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.65 17.65l1.42 1.42M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.65 6.35l1.42-1.42" /></svg>';
    themeToggle.innerHTML = icon;
    themeToggle.setAttribute("aria-label", `Switch to ${nextTheme} theme`);
    themeToggle.title = `Switch to ${nextTheme} theme`;
  };
  applyTheme(savedTheme || (prefersDark ? "dark" : "light"));
  themeToggle.addEventListener("click", () => {
    const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    try { localStorage.setItem("satquery-theme", theme); } catch { /* keep this session only */ }
    applyTheme(theme);
  });

  const updateScrollState = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.width = `${max > 0 ? (window.scrollY / max) * 100 : 0}%`;
    backToTop.classList.toggle("visible", window.scrollY > 900);
  };
  updateScrollState();
  window.addEventListener("scroll", updateScrollState, { passive: true });
  backToTop.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));

  if ("IntersectionObserver" in window) {
    const navLinks = [...sideToc.querySelectorAll("a")];
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (!visible.length) return;
      navLinks.forEach((link) => link.classList.toggle("active", link.dataset.target === visible[0].target.id));
    }, { rootMargin: "-18% 0px -72% 0px", threshold: 0 });
    parts.forEach((part) => observer.observe(part));
  }

  document.querySelectorAll("[data-figure]").forEach((figure) => {
    figure.tabIndex = 0;
    figure.setAttribute("role", "button");
    figure.setAttribute("aria-label", `${figure.querySelector("img").alt}. Open enlarged image.`);
    const openFigure = () => {
      const image = figure.querySelector("img");
      dialogImage.src = image.src;
      dialogImage.alt = image.alt;
      dialog.showModal();
    };
    figure.addEventListener("click", openFigure);
    figure.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openFigure();
      }
    });
  });
  dialogClose.addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
})();
