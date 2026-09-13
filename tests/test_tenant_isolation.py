from app.repositories.user import UserRepository


def test_user_from_org_a_not_returned_by_query_scoped_to_org_b(client, db_session, unique_email):
    email_a = unique_email("a")
    email_b = unique_email("b")

    resp_a = client.post(
        "/auth/register",
        json={"organization_name": "Org A", "email": email_a, "password": "supersecret1"},
    )
    resp_b = client.post(
        "/auth/register",
        json={"organization_name": "Org B", "email": email_b, "password": "supersecret1"},
    )

    org_a_id = resp_a.json()["user"]["organization_id"]
    org_b_id = resp_b.json()["user"]["organization_id"]
    user_a_id = resp_a.json()["user"]["id"]

    repo_scoped_to_b = UserRepository(db_session, organization_id=org_b_id)

    # The org-A user must not be visible through a repository scoped to org B,
    # whether looked up by id or by listing all users in org B.
    assert repo_scoped_to_b.get(user_a_id) is None
    assert all(str(u.id) != user_a_id for u in repo_scoped_to_b.list())

    repo_scoped_to_a = UserRepository(db_session, organization_id=org_a_id)
    assert repo_scoped_to_a.get(user_a_id) is not None
