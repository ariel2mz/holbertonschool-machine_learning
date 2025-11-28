#!/usr/bin/node

const request = require('request');
const movieId = process.argv[2];
const apiUrl = `https://swapi-api.alx-tools.com/api/films/${movieId}/`;

request(apiUrl, function (error, response, body) {
  if (error) return;
  const filmData = JSON.parse(body);
  const characterUrls = filmData.characters;

  function getCharacter(index) {
    if (index >= characterUrls.length) return;
    
    request(characterUrls[index], function (err, res, charBody) {
      if (err) return;
      console.log(JSON.parse(charBody).name);
      getCharacter(index + 1);
    });
  }

  getCharacter(0);
});