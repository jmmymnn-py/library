# app.py
import streamlit as st
import pandas as pd
import requests
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
import io
import base64
import json

# ----------------------------
# Config
# ----------------------------

# Local CSV fallback path (used when GitHub secrets are not set)
LOCAL_CSV_PATH = "export_0924_2025.csv"

# GitHub config from Streamlit secrets (if available)
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", None)
GITHUB_REPO = st.secrets.get("GITHUB_REPO", None)              # "owner/repo"
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")
GITHUB_FILE_PATH = st.secrets.get("GITHUB_FILE_PATH", None)    # e.g. "data/library.csv"
GITHUB_COMMITTER_NAME = st.secrets.get("GITHUB_COMMITTER_NAME", "Streamlit App")
GITHUB_COMMITTER_EMAIL = st.secrets.get("GITHUB_COMMITTER_EMAIL", "streamlit@app.local")

USE_GITHUB = bool(GITHUB_TOKEN and GITHUB_REPO and GITHUB_FILE_PATH)

# ----------------------------
# Utils: cover, saving, loading, authors
# ----------------------------

def get_default_cover_url(isbn13) -> str:
    """Return Open Library cover URL if ISBN13 present, else empty string."""
    if pd.isna(isbn13) or isbn13 in ("", None):
        return ""
    return f"https://covers.openlibrary.org/b/isbn/{isbn13}-L.jpg"

def _github_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

def _github_contents_url():
    owner_repo = GITHUB_REPO
    path = requests.utils.quote(GITHUB_FILE_PATH, safe="")
    return f"https://api.github.com/repos/{owner_repo}/contents/{path}"

def _github_read_csv() -> Optional[pd.DataFrame]:
    """Read CSV from GitHub via Contents API. Returns DataFrame or None on error."""
    try:
        params = {"ref": GITHUB_BRANCH}
        r = requests.get(_github_contents_url(), headers=_github_headers(), params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            content_b64 = data.get("content", "")
            decoded = base64.b64decode(content_b64).decode("utf-8")
            df = pd.read_csv(io.StringIO(decoded))
            return df
        elif r.status_code == 404:
            st.error(f"GitHub file not found: {GITHUB_REPO}@{GITHUB_BRANCH}:{GITHUB_FILE_PATH}")
        else:
            st.error(f"GitHub read failed: {r.status_code} {r.text}")
    except Exception as e:
        st.error(f"Error reading CSV from GitHub: {e}")
    return None

def _github_write_csv(df: pd.DataFrame, commit_message: str) -> bool:
    """Write CSV to GitHub via Contents API (creates a commit)."""
    try:
        params = {"ref": GITHUB_BRANCH}
        r_get = requests.get(_github_contents_url(), headers=_github_headers(), params=params, timeout=15)
        sha = None
        if r_get.status_code == 200:
            sha = r_get.json().get("sha")

        csv_str = df.to_csv(index=False)
        content_b64 = base64.b64encode(csv_str.encode("utf-8")).decode("utf-8")

        payload = {
            "message": commit_message,
            "content": content_b64,
            "branch": GITHUB_BRANCH,
            "committer": {
                "name": GITHUB_COMMITTER_NAME,
                "email": GITHUB_COMMITTER_EMAIL,
            },
        }
        if sha:
            payload["sha"] = sha

        r_put = requests.put(_github_contents_url(), headers=_github_headers(), data=json.dumps(payload), timeout=20)
        if r_put.status_code in (200, 201):
            return True
        else:
            st.error(f"GitHub write failed: {r_put.status_code} {r_put.text}")
            return False
    except Exception as e:
        st.error(f"Error writing CSV to GitHub: {e}")
        return False

@st.cache_data
def load_library() -> pd.DataFrame:
    """Load the library CSV and ensure required columns exist."""
    try:
        if USE_GITHUB:
            df = _github_read_csv()
            if df is None:
                df = pd.read_csv(LOCAL_CSV_PATH)
        else:
            df = pd.read_csv(LOCAL_CSV_PATH)
    except FileNotFoundError:
        loc = "GitHub" if USE_GITHUB else LOCAL_CSV_PATH
        st.error(f"CSV file not found ({loc}). Please ensure the file exists.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

    # Ensure required columns
    if "group" not in df.columns:
        df["group"] = "Unsorted"
    if "image_url" not in df.columns:
        df["image_url"] = ""
    if "date_added" not in df.columns:
        df["date_added"] = datetime.now().strftime("%Y-%m-%d")
    # ensure read_next column
    if "read_next" not in df.columns:
        df["read_next"] = False
    else:
        def _to_bool(x):
            s = str(x).strip().lower()
            if s in ("true", "1", "yes", "y", "t"):
                return True
            if s in ("false", "0", "no", "n", "f", "", "nan"):
                return False
            return False
        df["read_next"] = df["read_next"].apply(_to_bool)

    # Fill covers where missing
    missing_urls = df["image_url"].isna() | (df["image_url"].astype(str).str.strip() == "")
    if "ean_isbn13" in df.columns:
        df.loc[missing_urls, "image_url"] = df.loc[missing_urls, "ean_isbn13"].apply(get_default_cover_url)

    return df

def save_library(df: pd.DataFrame, reason: str = "Update library") -> None:
    """Save to GitHub or local CSV."""
    try:
        if USE_GITHUB:
            ok = _github_write_csv(df, commit_message=f"{reason} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if ok:
                st.success("Changes committed to GitHub ✅")
            else:
                st.error("Failed to commit changes to GitHub.")
        else:
            df.to_csv(LOCAL_CSV_PATH, index=False)
            st.success(f"Saved locally to {LOCAL_CSV_PATH} ✅")
    except Exception as e:
        st.error(f"Error saving CSV: {e}")

def get_single_book_row_by_title(df: pd.DataFrame, title: str) -> Optional[pd.Series]:
    """Return a single Series for the first row whose title matches title."""
    if df.empty or "title" not in df.columns:
        return None
    norm = df["title"].astype(str).str.strip().str.casefold()
    target = str(title).strip().casefold()
    matches = df[norm == target]
    if matches.empty:
        return None
    return matches.iloc[0]

def safe_str(val) -> str:
    if pd.isna(val) or val is None:
        return ""
    return str(val).strip()

def format_pages(val) -> str:
    """
    Return a pages string without trailing zeros if it's numeric;
    otherwise return the original string.
    """
    s = safe_str(val)
    if not s:
        return ""
    try:
        x = float(s.replace(",", ""))
        out = "{0:f}".format(x).rstrip("0").rstrip(".")
        return out or "0"
    except ValueError:
        return s

# ---------- Goodreads helpers ----------
def gr_search(q: str) -> str:
    return f"https://www.goodreads.com/search?q={requests.utils.quote(safe_str(q))}&search_type=books"

def goodreads_url_from_isbn(isbn: str) -> str:
    s = safe_str(isbn).replace("-", "").strip()
    return gr_search(s) if s else ""

# A tiny fallback "link button" for older Streamlit versions
def render_link_button(label: str, url: str):
    if not safe_str(url):
        return
    st.markdown(
        f"""
        <a href="{url}" target="_blank" rel="noopener"
           style="
             display:inline-block;
             padding:0.45rem 0.75rem;
             border-radius:0.5rem;
             background:#f0f2f6;
             text-decoration:none;
             font-weight:600;
             color:inherit;
             border:1px solid rgba(49,51,63,.2);
             ">
           {label}
        </a>
        """,
        unsafe_allow_html=True,
    )
# ---------------------------------------

def split_authors(creators_field):
    if pd.isna(creators_field) or not str(creators_field).strip():
        return ["Unknown"]
    creators_str = str(creators_field).strip()
    separators = [';', ',', '&', ' and ', '|']
    authors = [creators_str]
    for sep in separators:
        new_authors = []
        for author in authors:
            new_authors.extend([a.strip() for a in author.split(sep)])
        authors = new_authors
    authors = [a for a in authors if a and a.strip()]
    return authors if authors else ["Unknown"]

def get_books_by_individual_author(df: pd.DataFrame, target_author: str) -> pd.DataFrame:
    if df.empty or "creators" not in df.columns:
        return pd.DataFrame()
    target_lower = target_author.lower().strip()
    matching_indices = []
    for idx, row in df.iterrows():
        authors = split_authors(row.get("creators", ""))
        for author in authors:
            if author.lower().strip() == target_lower:
                matching_indices.append(idx)
                break
    return df.loc[matching_indices] if matching_indices else pd.DataFrame()

def get_query_param(key: str) -> str:
    q = st.query_params
    if key not in q:
        return ""
    value = q[key]
    if isinstance(value, list):
        return value[0] if value else ""
    return str(value)

# ----------------------------
# Third-party lookups for "Add Book"
# ----------------------------

def _fetch_open_library(isbn: str) -> Dict[str, Any]:
    try:
        r = requests.get(f"https://openlibrary.org/isbn/{isbn}.json", timeout=10)
        if r.status_code != 200:
            return {}
        data = r.json()
        title = data.get("title", "")
        publish_date = safe_str(data.get("publish_date", ""))
        pages = data.get("number_of_pages", "")
        authors_list = []
        for a in data.get("authors", []) or []:
            key = a.get("key")
            if not key:
                continue
            try:
                ar = requests.get(f"https://openlibrary.org{key}.json", timeout=6)
                if ar.status_code == 200:
                    authors_list.append(ar.json().get("name", ""))
            except Exception:
                pass
        authors = ", ".join([a for a in authors_list if a]) if authors_list else ""
        image_url = get_default_cover_url(isbn)
        description = ""
        desc = data.get("description", "")
        if isinstance(desc, dict):
            description = safe_str(desc.get("value", ""))
        else:
            description = safe_str(desc)
        return {
            "title": safe_str(title),
            "creators": safe_str(authors),
            "publish_date": safe_str(publish_date),
            "length": safe_str(pages),
            "image_url": image_url,
            "description": description,
            "ean_isbn13": safe_str(isbn),
        }
    except Exception:
        return {}

def _fetch_google_books(isbn: str) -> Dict[str, Any]:
    try:
        r = requests.get(f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}", timeout=10)
        if r.status_code != 200:
            return {}
        items = r.json().get("items", [])
        if not items:
            return {}
        vi = items[0].get("volumeInfo", {})
        title = vi.get("title", "")
        authors = ", ".join(vi.get("authors", []) or [])
        published_date = safe_str(vi.get("publishedDate", ""))
        page_count = safe_str(vi.get("pageCount", ""))
        img = (vi.get("imageLinks", {}) or {}).get("thumbnail", "")
        image_url = get_default_cover_url(isbn) or safe_str(img)
        description = safe_str(vi.get("description", ""))
        year = published_date[:4] if len(published_date) >= 4 and published_date[:4].isdigit() else published_date
        return {
            "title": safe_str(title),
            "creators": safe_str(authors),
            "publish_date": safe_str(year),
            "length": page_count,
            "image_url": image_url,
            "description": description,
            "ean_isbn13": safe_str(isbn),
        }
    except Exception:
        return {}

def fetch_book_by_isbn(isbn: str) -> Dict[str, Any]:
    isbn = safe_str(isbn).replace("-", "").strip()
    if not isbn:
        return {}
    data = _fetch_open_library(isbn)
    if data.get("title"):
        return data
    return _fetch_google_books(isbn)

# ----------------------------
# Libib Sync
# ----------------------------

def sync_with_libib(current_df: pd.DataFrame, libib_file) -> tuple[pd.DataFrame, dict]:
    try:
        libib_df = pd.read_csv(libib_file)
        isbn_columns = ['ean_isbn13', 'isbn13', 'isbn', 'ISBN', 'ISBN13', 'EAN']
        libib_isbn_col = next((c for c in isbn_columns if c in libib_df.columns), None)
        if libib_isbn_col is None:
            return current_df, {"error": "No ISBN column found in uploaded file"}

        existing_isbns = set(current_df["ean_isbn13"].dropna().astype(str)) if (not current_df.empty and "ean_isbn13" in current_df.columns) else set()
        libib_isbns = libib_df[libib_isbn_col].dropna().astype(str)
        new_books = libib_df[~libib_isbns.isin(existing_isbns)].copy()

        if new_books.empty:
            return current_df, {"added": 0, "skipped": len(libib_df), "message": "No new books found in upload"}

        column_mapping = {
            libib_isbn_col: "ean_isbn13",
            "title": "title",
            "creators": "creators",
            "authors": "creators",
            "author": "creators",
            "description": "description",
            "length": "length",
            "pages": "length",
            "page_count": "length",
            "publish_date": "publish_date",
            "publication_date": "publish_date",
            "published": "publish_date"
        }

        new_rows = []
        current_date = datetime.now().strftime("%Y-%m-%d")
        for _, book in new_books.iterrows():
            new_row = {
                "group": "New from Libib",
                "image_url": "",
                "date_added": current_date,
                "read_next": False,
            }
            for libib_col, our_col in column_mapping.items():
                if libib_col in book.index and not pd.isna(book[libib_col]):
                    new_row[our_col] = book[libib_col]
            new_rows.append(new_row)

        new_rows_df = pd.DataFrame(new_rows)
        updated_df = pd.concat([current_df, new_rows_df], ignore_index=True) if not current_df.empty else new_rows_df

        if "ean_isbn13" in updated_df.columns:
            missing_urls = updated_df["image_url"].isna() | (updated_df["image_url"].astype(str).str.strip() == "")
            updated_df.loc[missing_urls, "image_url"] = updated_df.loc[missing_urls, "ean_isbn13"].apply(get_default_cover_url)

        return updated_df, {"added": len(new_books), "skipped": len(libib_df) - len(new_books), "message": f"Successfully added {len(new_books)} new books"}
    except Exception as e:
        return current_df, {"error": f"Sync failed: {str(e)}"}

# ----------------------------
# Filtering / Sorting
# ----------------------------

def get_filter_options(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    options = {}
    if "group" in df.columns:
        options["groups"] = sorted(df["group"].fillna("Unsorted").astype(str).unique())
    if "creators" in df.columns:
        all_authors = []
        for creators in df["creators"].fillna("Unknown"):
            all_authors.extend(split_authors(creators))
        options["authors"] = sorted(list(set(all_authors)))
    if "publish_date" in df.columns:
        years = []
        for date in df["publish_date"].fillna(""):
            date_str = str(date)
            if len(date_str) >= 4 and date_str[:4].isdigit():
                years.append(date_str[:4])
        options["years"] = sorted(list(set(years)))
    return options

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if df.empty:
        return df
    filtered_df = df.copy()

    if filters.get("title_search"):
        search_term = str(filters["title_search"]).lower()
        title_match = filtered_df["title"].astype(str).str.lower().str.contains(search_term, na=False)
        if "description" in filtered_df.columns:
            desc_match = filtered_df["description"].astype(str).str.lower().str.contains(search_term, na=False)
        else:
            desc_match = pd.Series(False, index=filtered_df.index)
        filtered_df = filtered_df[title_match | desc_match]

    if filters.get("group") and filters["group"] != "All":
        filtered_df = filtered_df[filtered_df["group"].astype(str) == filters["group"]]

    if filters.get("author") and filters["author"] != "All":
        matching_indices = []
        for idx, row in filtered_df.iterrows():
            authors = split_authors(row.get("creators", ""))
            if filters["author"] in authors:
                matching_indices.append(idx)
        filtered_df = filtered_df.loc[matching_indices] if matching_indices else pd.DataFrame()

    if filters.get("year") and filters["year"] != "All":
        year_mask = filtered_df["publish_date"].astype(str).str.startswith(filters["year"])
        filtered_df = filtered_df[year_mask]

    return filtered_df

def sort_dataframe(df: pd.DataFrame, sort_by: str, ascending: bool = True) -> pd.DataFrame:
    if df.empty or sort_by not in df.columns:
        return df
    if sort_by == "publish_date":
        df_sorted = df.copy()
        df_sorted["sort_year"] = df_sorted["publish_date"].astype(str).str[:4]
        df_sorted["sort_year"] = pd.to_numeric(df_sorted["sort_year"], errors="coerce")
        df_sorted = df_sorted.sort_values("sort_year", ascending=ascending, na_position="last")
        return df_sorted.drop("sort_year", axis=1)
    elif sort_by == "date_added":
        return df.sort_values("date_added", ascending=ascending, na_position="last")
    else:
        return df.sort_values(sort_by, ascending=ascending, na_position="last")

# ----------------------------
# Views / UI Blocks
# ----------------------------

def display_library_controls(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Always-on search + expanders for Filters, Sort, Sync. Returns filtered/sorted df."""
    filter_options = get_filter_options(df)

    # Always-visible search + quick actions
    st.markdown('<a name="top"></a>', unsafe_allow_html=True)
    top_bar = st.container()
    with top_bar:
        c1, c2 = st.columns([3, 1])
        with c1:
            title_search = st.text_input("🔎 Search", placeholder="Title or description...", key="search_input")
        with c2:
            if st.button("➕ Add Book", use_container_width=True):
                st.session_state["navigate_to"] = ("add", "")

    # Expanders row: Filters / Sort / Sync
    e1, e2, e3 = st.columns(3)

    with e1:
        with st.expander("Filters", expanded=False):
            group_filter = st.selectbox("Group", ["All"] + filter_options.get("groups", []), key="flt_group")
            author_filter = st.selectbox("Author", ["All"] + filter_options.get("authors", [])[:50], key="flt_author")
            year_filter = st.selectbox("Year", ["All"] + filter_options.get("years", []), key="flt_year")

    with e2:
        with st.expander("Sort", expanded=False):
            sort_options = {
                "title": "Title",
                "creators": "Author",
                "publish_date": "Year",
                "date_added": "Added",
                "group": "Group"
            }
            sort_by = st.selectbox("Sort by", list(sort_options.keys()),
                                   format_func=lambda x: sort_options[x], key="sort_by")
            ascending = st.selectbox("Order", ["Ascending", "Descending"], key="sort_order") == "Ascending"

    with e3:
        with st.expander("Sync", expanded=False):
            if st.button("Export CSV"):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    export_filename = f"library_export_{timestamp}.csv"
                    df.to_csv(export_filename, index=False)
                    st.success(f"Exported to {export_filename}")
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="Download",
                        data=csv_data,
                        file_name=export_filename,
                        mime="text/csv",
                        key="download_export"
                    )
                except Exception as e:
                    st.error(f"Export failed: {e}")
            if st.button("Sync Libib"):
                st.session_state["show_libib_sync"] = True

    filters = {
        "title_search": st.session_state.get("search_input", ""),
        "group": st.session_state.get("flt_group", "All"),
        "author": st.session_state.get("flt_author", "All"),
        "year": st.session_state.get("flt_year", "All"),
    }

    filtered_df = apply_filters(df, filters)
    sorted_df = sort_dataframe(
        filtered_df,
        st.session_state.get("sort_by", "title"),
        st.session_state.get("sort_order", "Ascending") == "Ascending"
    )

    if len(sorted_df) != len(df):
        st.info(f"Showing {len(sorted_df)} of {len(df)} books")

    return sorted_df, filters

def display_libib_sync():
    st.header("📤 Sync with Libib")
    st.markdown("""
    **How to sync:**
    1. Export your library from Libib app as CSV  
    2. Upload the file below  
    3. New books (by ISBN) will be added to your library  
    4. Existing books remain unchanged
    """)
    uploaded_file = st.file_uploader(
        "Choose Libib CSV file",
        type=['csv'],
        help="Upload the CSV file exported from Libib"
    )

    if uploaded_file is not None:
        if st.button("🔄 Start Sync", type="primary"):
            current_df = load_library()
            updated_df, stats = sync_with_libib(current_df, uploaded_file)

            if "error" in stats:
                st.error(f"❌ {stats['error']}")
            else:
                save_library(updated_df, reason="Libib sync")
                st.cache_data.clear()

                st.success(f"✅ {stats['message']}")
                st.info(f"📈 Added: {stats['added']} books, Skipped: {stats['skipped']} existing books")

                if stats['added'] > 0:
                    st.subheader("📚 Newly Added Books:")
                    new_books = updated_df[updated_df["group"] == "New from Libib"]
                    for _, book in new_books.tail(10).iterrows():
                        st.write(f"• **{safe_str(book.get('title', 'Unknown'))}** by {safe_str(book.get('creators', 'Unknown'))}")

    if st.button("⬅ Back to Library"):
        st.session_state["show_libib_sync"] = False
        st.rerun()

def _display_read_next_strip(df: pd.DataFrame) -> None:
    """Show 'Read Next' list if any, under Featured Book and above Library."""
    if df.empty or "read_next" not in df.columns:
        return
    rn = df[df["read_next"] == True]
    if rn.empty:
        return

    st.subheader("📚 Read Next")
    cols = st.columns(3)
    for i, (idx, row) in enumerate(rn.iterrows()):
        col = cols[i % 3]
        with col:
            card = st.container(border=True)
            with card:
                cover_url = safe_str(row.get("image_url", "")) or get_default_cover_url(row.get("ean_isbn13"))
                if cover_url:
                    try:
                        st.image(cover_url, width=160)
                    except Exception:
                        st.write("📖")

                title = safe_str(row.get("title", "Untitled"))
                year_str = safe_str(row.get("publish_date", ""))
                year = year_str[:4] if year_str[:4].isdigit() else "N/A"
                group = safe_str(row.get("group", "Unsorted"))

                st.markdown(f"**{title}**")
                st.caption(f"Published: {year}")
                st.caption(f"Shelf: {group}")

                c_open, _ = st.columns(2)
                with c_open:
                    if st.button("Open", key=f"rn_open_{idx}", use_container_width=True):
                        st.session_state["navigate_to"] = ("book", title)

def display_featured_book(df: pd.DataFrame) -> None:
    """Display a random featured book, then the Read Next strip (if any), then the library grid."""
    if df.empty:
        st.warning("No books found in your library.")
        return

    featured_book = df.sample(n=1).iloc[0]
    st.subheader("Featured Book")

    col1, col2 = st.columns([2, 3])
    with col1:
        cover_url = safe_str(featured_book.get("image_url", "")) or get_default_cover_url(featured_book.get("ean_isbn13"))
        if cover_url:
            try:
                st.image(cover_url, width=300)
            except Exception:
                st.write("No cover available")
        else:
            st.write("No cover available")
    with col2:
        title = safe_str(featured_book.get("title", "Untitled"))
        st.markdown(f"# {title}")
        authors = safe_str(featured_book.get("creators", "Unknown"))
        pub_date = safe_str(featured_book.get("publish_date", ""))
        pub_year = pub_date[:4] if len(pub_date) >= 4 and pub_date[:4].isdigit() else "Unknown"
        group = safe_str(featured_book.get("group", "Unsorted"))
        pages = safe_str(featured_book.get("length", "Unknown"))
        st.markdown(f"**Author:** {authors}")
        st.markdown(f"**Published:** {pub_year}")
        st.markdown(f"**Group:** {group}")
        st.markdown(f"**Pages:** {pages}")
        description = safe_str(featured_book.get("description", ""))
        if description:
            preview = description[:200] + "..." if len(description) > 200 else description
            st.markdown(f"*{preview}*")

    _display_read_next_strip(df)

    st.divider()
    st.title("📚 My Library")

    if st.session_state.get("show_libib_sync", False):
        display_libib_sync()
        return

    display_df, _ = display_library_controls(df)
    if display_df.empty:
        st.warning("No books match your current filters.")
        return

    cols = st.columns(3)
    for i, (idx, row) in enumerate(display_df.iterrows()):
        col = cols[i % 3]
        with col:
            container = st.container(border=True)
            with container:
                cover_url = safe_str(row.get("image_url", "")) or get_default_cover_url(row.get("ean_isbn13"))
                if cover_url:
                    try:
                        st.image(cover_url, width=180)
                    except Exception:
                        st.write("📖 No cover")

                title = safe_str(row.get("title", "Untitled"))
                pub_date = safe_str(row.get("publish_date", ""))
                pub_year = pub_date[:4] if len(pub_date) >= 4 and pub_date[:4].isdigit() else "N/A"

                st.markdown(f"**{title}**")
                st.markdown(f":blue[Published: {pub_year}]")

                authors = split_authors(row.get("creators", "Unknown"))
                displayed_authors = authors[:2]
                remaining_count = len(authors) - 2

                if len(displayed_authors) == 1:
                    st.markdown("**Author:**")
                    if st.button(displayed_authors[0], key=f"author_{idx}_0"):
                        st.session_state["navigate_to"] = ("author", displayed_authors[0])
                else:
                    st.markdown("**Authors:**")
                    for j, author in enumerate(displayed_authors):
                        if st.button(author, key=f"author_{idx}_{j}"):
                            st.session_state["navigate_to"] = ("author", author)

                if remaining_count > 0:
                    st.markdown(f"*+{remaining_count} more author{'s' if remaining_count > 1 else ''}*")

                if st.button("📖 View Details", key=f"details_{idx}", use_container_width=True):
                    st.session_state["navigate_to"] = ("book", title)

    st.markdown('[⬆️ Back to top](#top)')

def display_library(df: pd.DataFrame) -> None:
    display_featured_book(df)

# ----------------------------
# Book Details (with left-side Group/Location + Goodreads under title)
# ----------------------------

def display_book_details(df: pd.DataFrame, book_title: str) -> None:
    if st.button("⬅ Back to Library"):
        st.session_state["navigate_to"] = ("home", "")

    book = get_single_book_row_by_title(df, book_title)
    if book is None:
        st.error(f"Could not find a book titled: {book_title!r}")
        if st.button("Go Home"):
            st.session_state["navigate_to"] = ("home", "")
        return

    st.header("📖 Book Details")
    book_idx = book.name

    # Global edit toggle (default OFF) — controls right-side fields & save button
    edit_mode = st.toggle("Edit", value=False, key=f"edit_mode_{book_idx}")

    # Current values
    cur_title = safe_str(book.get("title", ""))
    cur_creators = safe_str(book.get("creators", ""))
    cur_length = safe_str(book.get("length", ""))
    cur_publish_date = safe_str(book.get("publish_date", ""))
    cur_group = safe_str(book.get("group", "Unsorted"))
    cur_isbn = safe_str(book.get("ean_isbn13", ""))
    current_image_url = safe_str(book.get("image_url", ""))

    # Group list; ensure current group is present
    all_groups = df["group"].fillna("Unsorted").astype(str).unique().tolist()
    all_groups = sorted([g for g in all_groups if g])
    if cur_group and cur_group not in all_groups:
        all_groups.append(cur_group)
        all_groups = sorted(all_groups)

    col1, col2 = st.columns([2, 3])

    # -------- Left column: Cover + ALWAYS-editable Image URL + ALWAYS-editable Group/Location
    with col1:
        st.subheader("Cover Image")
        display_url = current_image_url or get_default_cover_url(book.get("ean_isbn13"))
        if display_url:
            try:
                st.image(display_url, width=200)
            except Exception:
                st.write("📖 No cover available")

        # Image URL — always editable
        new_image_url = st.text_input("Image URL", value=current_image_url, key=f"image_url_input_{book_idx}")

        # Group/Location — always editable, placed below Image URL
        try:
            group_index = all_groups.index(cur_group) if cur_group in all_groups else 0
        except ValueError:
            group_index = 0
        new_group = st.selectbox("Group/Location", all_groups, index=group_index, key=f"group_select_left_{book_idx}")

    # -------- Right column: Book Information (no Group/Location here)
    with col2:
        st.subheader("Book Information")

        if edit_mode:
            new_title = st.text_input("Title", value=cur_title, key=f"title_input_{book_idx}")

            # Goodreads button under Title (uses current ISBN in edit mode preview)
            gr_url_book_preview = goodreads_url_from_isbn(cur_isbn)
            render_link_button("Goodreads", gr_url_book_preview)

            new_creators = st.text_input("Authors/Creators", value=cur_creators, key=f"creators_input_{book_idx}")
            new_length = st.text_input("Page Count", value=cur_length, key=f"length_input_{book_idx}")
            new_publish_date = st.text_input("Publish Date", value=cur_publish_date, key=f"publish_date_input_{book_idx}")
            new_isbn = st.text_input("ISBN-13", value=cur_isbn, key=f"isbn_input_{book_idx}")

            # Optional Goodreads under ISBN as well
            gr_url_book_isbn = goodreads_url_from_isbn(new_isbn or cur_isbn)
            render_link_button("Goodreads (by ISBN)", gr_url_book_isbn)

        else:
            # Read-only view
            st.markdown(f"**Title:** {cur_title or 'Untitled'}")

            # Goodreads button directly under Title (uses ISBN)
            gr_url_book = goodreads_url_from_isbn(cur_isbn)
            render_link_button("Goodreads", gr_url_book)

            pages_display = format_pages(cur_length)
            st.markdown(f"**Creator:** {cur_creators or 'Unknown'}")
            year_txt = cur_publish_date[:4] if len(cur_publish_date) >= 4 and cur_publish_date[:4].isdigit() else (cur_publish_date or 'Unknown')
            st.markdown(f"**Published:** {year_txt or 'Unknown'}")
            st.markdown(f"**Pages:** {pages_display or 'Unknown'}")
            st.markdown(f"**ISBN-13:** {cur_isbn or '(none)'}")

            # carry forward for save
            new_title = cur_title
            new_creators = cur_creators
            new_length = cur_length
            new_publish_date = cur_publish_date
            new_isbn = cur_isbn

        # Quick Author Search (always available)
        st.markdown("**Quick Author Search:**")
        authors = split_authors(book.get("creators", "Unknown"))
        for i, author in enumerate(authors):
            if st.button(f"🔍 {author}", key=f"search_author_{book_idx}_{i}"):
                st.session_state["navigate_to"] = ("author", author)

    # Read Next toggle (remains live)
    st.subheader("Read Next")
    current_read_next = bool(book.get("read_next", False))
    new_read_next = st.toggle("Add this book to your Read Next list", value=current_read_next, key=f"read_next_toggle_{book_idx}")
    if new_read_next != current_read_next:
        try:
            df.at[book_idx, "read_next"] = new_read_next
            save_library(df, reason="Toggle Read Next")
            st.cache_data.clear()
            st.success("✅ Read Next updated.")
        except Exception as e:
            st.error(f"❌ Error updating Read Next: {e}")

    # Description (gated by edit mode)
    st.subheader("Description")
    current_description = safe_str(book.get("description", ""))
    if edit_mode:
        new_description = st.text_area("Description", value=current_description, height=150, key=f"description_input_{book_idx}")
    else:
        if current_description:
            st.markdown(current_description)
        else:
            st.caption("No description.")
        new_description = current_description

    st.divider()

    # Save button (saves both left + right edits) only visible in edit mode
    if edit_mode:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button("💾 Save All Changes", key=f"save_all_button_{book_idx}", type="primary"):
                try:
                    df.at[book_idx, "title"] = new_title
                    df.at[book_idx, "creators"] = new_creators
                    df.at[book_idx, "length"] = new_length
                    df.at[book_idx, "publish_date"] = new_publish_date
                    df.at[book_idx, "group"] = new_group             # ← save from left
                    df.at[book_idx, "image_url"] = new_image_url      # ← save from left
                    df.at[book_idx, "description"] = new_description
                    if "ean_isbn13" in df.columns:
                        df.at[book_idx, "ean_isbn13"] = new_isbn

                    old_title = cur_title
                    save_library(df, reason="Edit book details")
                    st.cache_data.clear()
                    st.success("✅ All changes saved successfully!")

                    if new_title != old_title:
                        st.info("Title was changed. Redirecting...")
                        st.session_state["navigate_to"] = ("book", new_title)
                except Exception as e:
                    st.error(f"❌ Error saving changes: {e}")

    with st.expander("📋 Current Values (Reference)", expanded=False):
        st.write("**Original Title:**", safe_str(book.get("title", "")))
        st.write("**Original Authors:**", safe_str(book.get("creators", "")))
        st.write("**Original Group:**", safe_str(book.get("group", "")))
        st.write("**Date Added:**", safe_str(book.get("date_added", "")))

    st.markdown('[⬆️ Back to top](#top)')

def display_author_books(df: pd.DataFrame, author: str) -> None:
    if st.button("⬅ Back to Library"):
        st.session_state["navigate_to"] = ("home", "")
    st.header(f"📖 Books by {author}")

    # Goodreads author search "button"
    render_link_button(f"Goodreads: {author}", gr_search(author))

    if not df.empty:
        matches = get_books_by_individual_author(df, author)
        st.subheader("In Your Library")
        if matches.empty:
            st.info(f"No books by '{author}' found in your library.")
        else:
            st.write(f"Found {len(matches)} book(s) by {author}:")
            for idx, row in matches.iterrows():
                title = safe_str(row.get("title", "Untitled"))
                all_authors = safe_str(row.get("creators", "Unknown"))
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{title}**")
                    if all_authors != author:
                        st.write(f"*Co-authors: {all_authors}*")
                with col2:
                    if st.button("View", key=f"auth_owned_{idx}"):
                        st.session_state["navigate_to"] = ("book", title)

    st.subheader("Other Books by This Author (Open Library)")
    try:
        url = f"https://openlibrary.org/search.json?author={requests.utils.quote(author)}&limit=30"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            docs = data.get("docs", [])
            if not docs:
                st.info("No additional books found on Open Library.")
            else:
                for i, doc in enumerate(docs[:30]):
                    title = safe_str(doc.get("title", "Unknown Title"))
                    cover_id = doc.get("cover_i")
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if cover_id:
                                cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"
                                try:
                                    st.image(cover_url, width=80)
                                except Exception:
                                    st.write("📖")
                            else:
                                st.write("📖")
                        with col2:
                            st.write(f"**{title}**")
                            pub_year = doc.get("first_publish_year", "N/A")
                            if pub_year and pub_year != "N/A":
                                st.write(f"Published: {pub_year}")
    except requests.RequestException as e:
        st.warning(f"Could not connect to Open Library: {e}")
    except Exception as e:
        st.error(f"Error fetching from Open Library: {e}")

    st.markdown('[⬆️ Back to top](#top)')

def display_group_books(df: pd.DataFrame, group: str) -> None:
    if st.button("⬅ Back to Library"):
        st.session_state["navigate_to"] = ("home", "")
    st.header(f"🛋 Books in: {group}")

    if df.empty or "group" not in df.columns:
        st.info("No books found.")
        return

    matches = df[df["group"].astype(str).str.strip() == group.strip()]
    if matches.empty:
        st.info(f"No books found in the '{group}' group.")
    else:
        for idx, row in matches.iterrows():
            title = safe_str(row.get("title", "Untitled"))
            if st.button(title, key=f"group_{idx}"):
                st.session_state["navigate_to"] = ("book", title)

    st.markdown('[⬆️ Back to top](#top)')

# ----------------------------
# Add Book Page
# ----------------------------

def display_add_book(df: pd.DataFrame) -> None:
    if st.button("⬅ Back to Library"):
        st.session_state["navigate_to"] = ("home", "")

    st.header("➕ Add a Book by ISBN")
    isbn = st.text_input("ISBN-13 (hyphens ok)", key="add_isbn_input", placeholder="9780140177398")
    if st.button("Search ISBN", type="primary"):
        st.session_state["add_result"] = fetch_book_by_isbn(isbn)

    result = st.session_state.get("add_result", {})
    if result:
        st.subheader("Result")
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                cover_url = safe_str(result.get("image_url", "")) or get_default_cover_url(result.get("ean_isbn13", ""))
                if cover_url:
                    try:
                        st.image(cover_url, width=150)
                    except Exception:
                        st.write("📖")
            with c2:
                title = st.text_input("Title", value=safe_str(result.get("title", "")), key="add_title")
                creators = st.text_input("Authors/Creators", value=safe_str(result.get("creators", "")), key="add_creators")
                publish_date = st.text_input("Publish Year", value=safe_str(result.get("publish_date", "")), key="add_publish_date")
                length = st.text_input("Page Count", value=safe_str(result.get("length", "")), key="add_length")
                image_url = st.text_input("Image URL", value=safe_str(result.get("image_url", "")), key="add_image_url")
                description = st.text_area("Description", value=safe_str(result.get("description", "")), key="add_description", height=120)
                ean_isbn13 = st.text_input("ISBN-13", value=safe_str(result.get("ean_isbn13", isbn)), key="add_ean")

                groups = sorted([g for g in df.get("group", pd.Series([])).fillna("Unsorted").astype(str).unique().tolist() if g])
                group_choice = st.selectbox("Shelf / Group (optional)", ["(none)"] + groups, key="add_group")

                if st.button("💾 Add to Library", type="primary"):
                    if not ean_isbn13:
                        st.error("ISBN-13 is required to add.")
                        return

                    if not df.empty and "ean_isbn13" in df.columns:
                        if ean_isbn13 in set(df["ean_isbn13"].dropna().astype(str)):
                            st.warning("This ISBN already exists in your library.")
                            return

                    new_row = {
                        "title": safe_str(title) or "Untitled",
                        "creators": safe_str(creators),
                        "publish_date": safe_str(publish_date),
                        "length": safe_str(length),
                        "image_url": safe_str(image_url),
                        "description": safe_str(description),
                        "ean_isbn13": safe_str(ean_isbn13),
                        "group": (group_choice if group_choice != "(none)" else "Unsorted"),
                        "date_added": datetime.now().strftime("%Y-%m-%d"),
                        "read_next": False,
                    }
                    updated_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True) if not df.empty else pd.DataFrame([new_row])
                    save_library(updated_df, reason="Add book (manual)")
                    st.cache_data.clear()
                    st.success("✅ Book added!")
                    st.session_state.pop("add_result", None)
                    st.session_state["navigate_to"] = ("book", new_row["title"])

    st.markdown('[⬆️ Back to top](#top)')

# ----------------------------
# Main App and Navigation
# ----------------------------

def handle_navigation():
    if "navigate_to" in st.session_state:
        nav_type, nav_value = st.session_state["navigate_to"]
        st.query_params.clear()
        if nav_type == "book":
            st.query_params["book"] = nav_value
        elif nav_type == "author":
            st.query_params["author"] = nav_value
        elif nav_type == "group":
            st.query_params["group"] = nav_value
        elif nav_type == "add":
            st.query_params["add"] = "1"
        del st.session_state["navigate_to"]

def main() -> None:
    handle_navigation()
    df = load_library()

    with st.sidebar:
        if st.button("➕ Add Book", use_container_width=True, key="side_add"):
            st.session_state["navigate_to"] = ("add", "")
        st.subheader("Debug Info")
        if st.checkbox("Show Debug", value=False):
            st.write("Query Params:", dict(st.query_params))
            st.write("Session State:", {k: v for k, v in st.session_state.items() if not k.startswith('_')})
            st.write("DataFrame shape:", df.shape if not df.empty else "Empty")
            st.write("Storage mode:", "GitHub" if USE_GITHUB else "Local CSV")

    book_param = get_query_param("book")
    author_param = get_query_param("author")
    group_param = get_query_param("group")
    add_param = get_query_param("add")

    if add_param:
        display_add_book(df)
    elif book_param:
        display_book_details(df, book_param)
    elif author_param:
        display_author_books(df, author_param)
    elif group_param:
        display_group_books(df, group_param)
    else:
        display_library(df)

if __name__ == "__main__":
    main()
