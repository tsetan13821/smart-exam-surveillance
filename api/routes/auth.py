# api/routes/auth.py
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from api import auth_store
from api.deps import current_user, get_settings
from api.schemas import LoginRequest, User
from api.security import verify_password_or_dummy

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=User)
def login(body: LoginRequest, request: Request, response: Response, settings=Depends(get_settings)):
    throttle = request.app.state.login_throttle
    client = request.client.host if request.client else "unknown"
    key = f"{body.username.lower()}|{client}"
    wait = throttle.retry_after(key)
    if wait:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            f"Too many failed logins. Try again in {wait} s.",
                            headers={"Retry-After": str(wait)})

    user = auth_store.get_user_by_username(body.username)
    if not verify_password_or_dummy(body.password, user["password_hash"] if user else None):
        throttle.record_failure(key)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid username or password.")

    throttle.reset(key)
    # A fresh token on every login, so a token planted before login can't be reused (session fixation)
    old = request.cookies.get(settings.cookie_name)
    if old:
        auth_store.delete_login_session(old)
    token = auth_store.create_login_session(user["user_id"], settings.session_hours)
    response.set_cookie(settings.cookie_name, token, max_age=int(settings.session_hours * 3600),
                        httponly=True, samesite="strict", secure=settings.cookie_secure, path="/")
    return {"username": user["username"]}


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, response: Response, settings=Depends(get_settings)):
    auth_store.delete_login_session(request.cookies.get(settings.cookie_name))
    response.delete_cookie(settings.cookie_name, path="/", httponly=True, samesite="strict",
                           secure=settings.cookie_secure)


@router.get("/me", response_model=User)
def me(user=Depends(current_user)):
    return {"username": user["username"]}
