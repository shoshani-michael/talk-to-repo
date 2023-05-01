import React, { useState } from "react";

function GitHubInput() {
  const [username, setUsername] = useState("");
  const [repo, setRepo] = useState("");

  const handleClick = async () => {
    try {
      const response = await fetch("http://localhost:8000/load_repo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, repo }),
      });
      const data = await response.json();
      console.log(data);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div>
      <label>
        GitHub Username:
        <input
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
      </label>
      <br />
      <label>
        Repo:
        <input
          type="text"
          value={repo}
          onChange={(e) => setRepo(e.target.value)}
        />
      </label>
      <br />
      <button onClick={handleClick}>
        Load Repo
      </button>
    </div>
  );
}

export default GitHubInput;