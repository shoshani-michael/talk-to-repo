import React from 'react';
import GitHubInput from './GitHubInput';
const Header = (props) => {
  return (
    <div className="bg-gray-700 text-white text-center py-4">
      <h1 className="text-2xl font-semibold">Talk to Repo</h1>
      <GitHubInput
        importMessages={props.importMessages}
        clearMessages={props.clearMessages}        messages={props.messages}
        handleCommitCodeSnippets={props.handleCommitCodeSnippets}
      />
      <div className="pt-2 text-center text-sm text-gray-400">
        <a
          href="https://github.com/twitter/the-algorithm"
          target="_blank"          rel="noopener noreferrer"
        >
          based on https://github.com/twitter/the-algorithm
        </a>
      </div>
    </div>
  );
};

export default Header;