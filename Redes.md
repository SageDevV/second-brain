
## Conceitos

1. **Camada Física**: Esta camada lida com a transmissão e recepção de sinais brutos não estruturados sobre um meio físico. Ela inclui especificações como níveis de tensão, taxas de transmissão de dados, comprimentos máximos de cabos, etc. A analogia seria o carteiro que entrega a carta. Ele não se importa com o conteúdo da carta, apenas com a entrega física dela.

2. **Camada de Enlace de Dados**: Esta camada transforma o meio físico bruto em uma linha que aparece livre de erros de transmissão não detectados para a camada de rede. Ela fornece o controle de fluxo e sequenciamento de pacotes. A analogia seria o envelope que envolve a carta. Ele garante que a carta chegue ao destinatário corretamente e sem danos.

3. **Camada de Rede**: A camada de rede controla a operação da sub-rede, decidindo qual caminho físico os dados devem seguir com base nas condições da rede, prioridade do serviço e outros fatores. A analogia seria o endereço na carta. Ele decide para onde a carta deve ir.

4. **Camada de Transporte**: A camada de transporte garante que as mensagens sejam entregues sem erros, na sequência correta e sem perdas ou duplicações. Ela alivia a camada de sessão de qualquer preocupação com a transferência de dados entre eles e seus pares. A analogia seria a verificação de que a carta foi entregue corretamente. Se a carta se perder, esta camada se encarrega de enviar outra.

5. **Camada de Sessão**: A camada de sessão permite que duas aplicações em computadores diferentes estabeleçam, usem e encerrem uma conexão, chamada sessão. A analogia seria marcar um encontro para entregar a carta pessoalmente. Ela estabelece, gerencia e termina a conexão entre o remetente e o destinatário.

6. **Camada de Apresentação**: Esta camada fornece independência das diferenças de representação de dados (por exemplo, criptografia) para permitir que a camada de aplicação leia os dados como se fossem do mesmo sistema. A analogia seria o idioma em que a carta é escrita. Ela garante que o destinatário possa entender a mensagem.

7. **Camada de Aplicação**: Esta é a camada que interage diretamente com o software que está usando a rede. Ela fornece serviços de rede para aplicativos de software. A analogia seria o conteúdo da carta em si. É a informação que o remetente quer transmitir para o destinatário.


## Comandos

1. **Comando Ping:** O comando `ping` envia pacotes ICMP Echo Request para um destino especificado e aguarda uma resposta. Ele mede o tempo que leva para a resposta chegar, que é o tempo de ida e volta.

	- `-t`: Ping continuado. O ping será enviado ao destino até que o usuário interrompa o comando com Ctrl+C.
	- `-a`: Resolve endereços para nomes de host.
	- `-n count`: Número de solicitações de eco a enviar.
	- `-l size`: Tamanho do buffer de envio.

	**Analogia do Ping:** Você pode pensar no comando `ping` como jogar uma bola contra uma parede e medir o tempo que leva para a bola voltar para você. Se a bola voltar rapidamente, isso significa que a parede (o destino) está próxima. Se demorar, a parede está mais distante.

2. **Comando Tracert:** O comando `tracert` é usado para determinar a rota que um pacote leva para chegar ao seu destino. Ele faz isso enviando pacotes com valores TTL (Time to Live) incrementais, começando com 1.

	Aqui estão algumas das opções técnicas do comando `tracert`:

	- `-d`: Evita a resolução de endereços IP em nomes de host.
	- `-h MaxHops`: Número máximo de saltos para pesquisar o destino.
	- `-w TimeOut`: Tempo de espera em milissegundos para cada resposta.
	
	**Analogia do Tracert:** Você pode pensar no comando `tracert` como fazer uma viagem de carro de uma cidade para outra. Em cada cidade que você passa (cada “salto”), você anota o nome da cidade e quanto tempo levou para chegar lá desde a última cidade. No final, você terá uma lista completa de todas as cidades pelas quais passou e quanto tempo levou para passar por cada uma delas.



