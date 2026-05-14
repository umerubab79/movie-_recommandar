import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
from webbrowser import get
import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY is missing. put it in .env as TMDB_API_KEY=your_tmdb_api_key_here")


app = FastAPI(title="Movie Recommendation API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#path and global vars cofiguration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DF_PATH = os.path.join(BASE_DIR,"df.pkl")
INDICES_PATH = os.path.join(BASE_DIR,"indices.pkl") 
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR,"tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR,"tfidf.pkl") 

df: Optional[pd.DataFrame] = None
inices_obj: Any = None
tfidf_matrix: any = None
tfidf_obj: any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None

# models
class TMDBMovie(BaseModel):
    tmbd_id: int
    title: str
    overview: Optional[str] = None
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
        tmbd_id: int
        title: str
        overview: Optional[str] = None 
        release_date: Optional[str] = None   
        poster_url: Optional[str] = None
        backdrop_url: Optional[str] = None
        genres: Optional[List[str]] = []


class TMDBMovieCard(BaseModel):
    tmbd_id: int
    title: str
    overview: Optional[str] = None
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None



class TFIDFRecItem(BaseModel):
    title: str
    score : float
    tmdb : Optional[TMDBMovie] = None



class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovie]

#utlity functions

def _norm_title(t: str) -> str:
    return t.strip().lower()



def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    q = dict (params)
    q["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE_URL}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
              detail=f"TMDB request error: {type(e).__name__}  | - {repr(e)}",

              )
    

    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB error: {r.status_code} | - {r.text}",
        )
    
    return r.json()



async def tmdb_get_cards_from_results(
        results: List[Dict], limit : int = 20 
        ) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in results[:limit]:
        out.append(
             TMDBMovieCard(
                tmbd_id=(m["id"]),
                title=m["title"],
                overview=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out



async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}" , {'language': 'en-US'})
    return TMDBMovieDetails(
        tmbd_id=data["id"],
        title=data["title"],
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=[g["name"] for g in data.get("genres", [])] or [],
    )

async def tmdb_search_movie(query: str, page: int = 1) -> Dict[str, Any]:
    return await tmdb_get(
        "/search/movie",
          {"query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
            }
            )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movie(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


def build_title_to_idx_map(indices:Any) -> Dict[str, int]:
    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
            return title_to_idx
        
        
  
        

    #pandas series or similar mapping
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
            return title_to_idx
    except Exception:
        raise RuntimeError(
            " indices.pkl must be dict or pandas Series-like(with .item())"

        )
            
def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TITLE_TO_IDX is not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(status_code=404,
                         detail=f"Movie title '{title}' not found in local dataset"
                         )
    
    return TITLE_TO_IDX.get(_norm_title(title))


def tfidf_recomand_title(
        quary_title: str, top_k: int = 10
        ) -> List[Tuple[str, float]]:
    #return list of (title, score) tuples 
    global df,tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500,
                             detail="Data not loaded")
    
    idx = get_local_idx_by_title(quary_title)

    #quary_vec = tfidf_obj.transform([quary_title])
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    #sort descending and get top k

    order = np.argsort(scores)
    out: List[Tuple[str, float]] = []
    for i in order:
        if int (i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[i]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[i])))
        if len(out) >= top_k:
            break   
    return out

async def attach_tmbd_card_by_title(title:str) -> Optional[TMDBMovieCard]:
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmbd_id=m["id"],
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),

        )
    except Exception:
        return None
    

 # startup event to load pickles and build title to idx map
@app.on_event("startup")
def load_pickles():
    global df, inices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX    
    # load df
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

     #load indices
    with open(INDICES_PATH, "rb") as f:
        inices_obj = pickle.load(f)
    
    #load Tfidf matrix
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)


     #load tfidf vectorizer
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

     #load build normalized map 
    TITLE_TO_IDX = build_title_to_idx_map(inices_obj)

     #sanity
    if df is None or "title" not in df.columns:
        raise RuntimeError('df.pkl must contain a DataFrame with a "title" column')   



#routes
@app.get("/health")
def health():
    return {"status": "ok"}

# home routes
@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category:str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US", "page": 1})
            return await tmdb_get_cards_from_results(data.get("results", []), limit=limit)
        
        if category not in {"popular","top_rated","upcoming","now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")
        
        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        return await tmdb_get_cards_from_results(data.get("results", []), limit=limit)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching home data: {e}"
        )
# search miltiple via keywords
@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=1000)
):
  return await tmdb_search_movie(query=query, page=page)


#movie details + reccomendations bundle
@app.get("/movie/id/{tmdb_id}", response_model=SearchBundleResponse)
async def movie_details_bundle(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)
    #genre based reccomendations
@app.get("/recommand/genre", response_model=List[TMDBMovie])
async def recomand_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50)
):
    "given a tmdb movie id, return genre based reccomendations"
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []
    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie", 
        {"with_genres": genre_id,
          "language": "en-US", "page": 1})
    cards = await tmdb_get_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmbd_id != tmdb_id]

#tfidf based reccomendations
@app.get("/recomand_tfidf")
async def recomand_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),):
    recs = tfidf_recomand_title(title, top_n=top_n)
    return [{ "title": t, "score": s } for t, s in recs]


# bundle route for search + details + reccomendations
@app.get('/movie/search', response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"Movie not found in TMDB: {query}"

            )
    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)
    #tfidf recommand never crash endpoint
    tfidf_items: List[TFIDFRecItem] = []
    recs: List[Tuple[str, float]] = []
    try:
        recs = tfidf_recomand_title(details.title, top_n=tfidf_top_n)
    except Exception:
        recs = []
    for title, score in recs:
        card = await attach_tmbd_card_by_title(title)
        tfidf_items.append(
            TFIDFRecItem(
                title=title,
                score=score,
                tmdb=card
            )
        )
        #genre reccomendations tmdb discover by the first genre
    genre_recs: List[TMDBMovie] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie", 
            {
                "with_genres": genre_id,
              "language": "en-US",
               "sort_by": "popularity.desc",
               "page": 1})
        cards = await tmdb_get_cards_from_results(discover.get("results", []), limit=genre_limit)
        genre_recs = [c for c in cards if c.tmbd_id != tmdb_id]
    return SearchBundleResponse(
        query=query,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,)



    
    