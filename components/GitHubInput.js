import React, { useState } from "react";

function GitHubInput() {
  const [username, setUsername] = useState("");
  const [repo, setRepo] = useState("");

  const handleClick = async () => {
    try {
      const response = await fetch(process.env.NEXT_PUBLIC_BACKEND_URL + "/load_repo", {
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

    <div className="panel-container "
    style={{position: 'fixed', left: 0, top: 'calc(50% - 50px)'}}
    > {}
      <input
        type="text"
        value={username}
        placeholder="GitHub Username"
        className="flex-1 p-2 border rounded-lg focus:outline-none focus:ring focus:border-blue-300 resize-none overflow-hidden bg-gray-600 text-gray-100"        
        onChange={(e) => setUsername(e.target.value)}
      />
      <br />
      <input
        type="text"
        value={repo}
        placeholder="Repository Name"
        className="flex-1 p-2 border rounded-lg focus:outline-none focus:ring focus:border-blue-300 resize-none overflow-hidden bg-gray-600 text-gray-100"        
          onChange={(e) => setRepo(e.target.value)}
      />
      <br />
      <button onClick={handleClick} 
        className="ml-2 px-2 py-1 rounded-lg bg-blue-500 text-white focus:outline-none hover:bg-blue-600 md:ml-4 md:px-4 md:py-2">
        Load Repo
      </button>
    </div>
  );
}

export default GitHubInput;