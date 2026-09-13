import jwt

from app.config import get_settings
from app.models.organization import Organization
from app.models.user import User

settings = get_settings()


def test_register_creates_org_and_admin_user(client, db_session, unique_email):
    email = unique_email()
    response = client.post(
        "/auth/register",
        json={"organization_name": "Acme Inc", "email": email, "password": "supersecret1"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["user"]["email"] == email
    assert body["user"]["role"] == "admin"
    assert "access_token" in body

    org = db_session.query(Organization).filter_by(name="Acme Inc").one()
    user = db_session.query(User).filter_by(email=email).one()
    assert user.organization_id == org.id
    assert user.role.value == "admin"
    assert user.hashed_password != "supersecret1"


def test_login_with_correct_credentials_returns_valid_jwt(client, unique_email):
    email = unique_email()
    client.post(
        "/auth/register",
        json={"organization_name": "Acme Inc", "email": email, "password": "supersecret1"},
    )

    response = client.post("/auth/login", json={"email": email, "password": "supersecret1"})

    assert response.status_code == 200
    token = response.json()["access_token"]
    payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    assert "sub" in payload
    assert "organization_id" in payload
    assert payload["role"] == "admin"


def test_login_with_wrong_password_returns_401(client, unique_email):
    email = unique_email()
    client.post(
        "/auth/register",
        json={"organization_name": "Acme Inc", "email": email, "password": "supersecret1"},
    )

    response = client.post("/auth/login", json={"email": email, "password": "wrong-password"})

    assert response.status_code == 401
