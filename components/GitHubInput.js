import React, { useState, useEffect } from "react";

function GitHubInput(props) {
  const [username, setUsername] = useState("");
  const [repo, setRepo] = useState("");
  const [token, setToken] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState(null);
  
  useEffect(() => {
    const storedUsername = localStorage.getItem("username");
    const storedRepo = localStorage.getItem("repo");
    const storedToken = localStorage.getItem("token");
  
    if (storedUsername) setUsername(storedUsername);
    if (storedRepo) setRepo(storedRepo);
    if (storedToken) setToken(storedToken);
  }, []);

  const handleClick = async () => {
    setLoading(true); // Start loading
    setLoadingStatus(null);
    
    try {
      const response = await fetch(process.env.NEXT_PUBLIC_BACKEND_URL + "/load_repo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, repo, token }), 
      });

      const data = await response.json();
      console.log(data);
      
      setLoading(false); // Stop loading
      setLoadingStatus(response.ok ? 'success' : 'error');
      if (response.ok) {
        localStorage.setItem("username", username);
        localStorage.setItem("repo", repo);
        localStorage.setItem("token", token);
        props.clearMessages(); 
      }
    } catch (error) {
      console.error(error);
      setLoading(false); // Stop loading
      setLoadingStatus('error');
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
      <input
        type="text"
        value={token}
        placeholder="GitHub API Token"
        className="flex-1 p-2 border rounded-lg focus:outline-none focus:ring focus:border-blue-300 resize-none overflow-hidden bg-gray-600 text-gray-100"
        onChange={(e) => setToken(e.target.value)}
      />
      <br />
      <button onClick={handleClick} className="ml-2 px-2 py-1 rounded-lg bg-blue-500 text-white focus:outline-none hover:bg-blue-600 md:ml-4 md:px-4 md:py-2">
        Load Repo
      </button>
      {loading && <span className="ml-2 animate-spin">&#9203;</span>}
      {!loading && loadingStatus === 'success' && <span className="ml-2 text-green-500">&#10004;</span>}
      {!loading && loadingStatus === 'error' && <span className="ml-2 text-red-500">&#10006;</span>}
    </div>
  );
}

export default GitHubInput;