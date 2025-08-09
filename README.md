# 113bommy.github.io — Personal Blog & Portfolio

Welcome! This repository hosts my personal blog built with **Jekyll** and served via **GitHub Pages**.  
I use it to share technical notes, project write-ups, and experiments around AI, data science, and software.

---

## Quick start (local preview)

### 1. Prerequisites
- Ruby (≥ 3.0 recommended) and Bundler
- Optional: Node.js (for bundling/minifying assets if you add a pipeline)

### 2. Install
```bash
gem install bundler
bundle install
```

### 3. Run locally
```bash
bundle exec jekyll serve --livereload
# Open http://localhost:4000
```

### 4. Deploy
- Push to the default branch. Github Pages will build and publish automatically.
- For a custom domain, add **CNAME** file and set DNS to `username.github.io`

### Repository layout
```bash
.
├── _config.yml           # Site configuration
├── _posts/               # Blog posts (Markdown files)
├── _drafts/              # Optional: unpublished drafts
├── _pages/               # Optional: standalone pages
├── _includes/            # HTML includes
├── _layouts/             # HTML layouts
├── _sass/                # Theme styles
├── assets/
│   ├── img/              # Images
│   ├── css/              # Stylesheets
│   └── js/               # Scripts
└── README.md
```

### License
Content © 113bommy.
Theme and plugins under their respective licenses.
