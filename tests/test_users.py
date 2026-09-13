def test_users_me_without_token_returns_401(client):
    response = client.get("/users/me")
    assert response.status_code == 401


def test_users_me_with_valid_token_returns_correct_user(client, unique_email):
    email = unique_email()
    register_response = client.post(
        "/auth/register",
        json={"organization_name": "Acme Inc", "email": email, "password": "supersecret1"},
    )
    token = register_response.json()["access_token"]

    response = client.get("/users/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    body = response.json()
    assert body["email"] == email
    assert body["role"] == "admin"


def test_users_me_with_invalid_token_returns_401(client):
    response = client.get("/users/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401
